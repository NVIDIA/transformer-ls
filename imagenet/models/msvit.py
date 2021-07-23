"""Code for the vision transformer model based on ViL.
Adapted from https://github.com/microsoft/vision-longformer by Chen Zhu (zhuchen.eric@gmail.com)
"""
import math
from functools import partial
import logging
import torch
from torch import nn
from timm.models.layers import DropPath, trunc_normal_, to_2tuple
from .layers import Attention, AttentionLS


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size, nx, ny, in_chans=3, embed_dim=768, nglo=1,
                 norm_layer=nn.LayerNorm, norm_embed=True, drop_rate=0.0,
                 ape=True):
        # maximal global/x-direction/y-direction tokens: nglo, nx, ny
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=patch_size)

        self.norm_embed = norm_layer(embed_dim) if norm_embed else None

        self.nx = nx
        self.ny = ny
        self.Nglo = nglo
        if nglo >= 1:
            self.cls_token = nn.Parameter(torch.zeros(1, nglo, embed_dim))
            trunc_normal_(self.cls_token, std=.02)
        else:
            self.cls_token = None
        self.ape = ape
        if ape:
            self.cls_pos_embed = nn.Parameter(torch.zeros(1, nglo, embed_dim))
            self.x_pos_embed = nn.Parameter(torch.zeros(1, nx, embed_dim // 2))
            self.y_pos_embed = nn.Parameter(torch.zeros(1, ny, embed_dim // 2))
            trunc_normal_(self.cls_pos_embed, std=.02)
            trunc_normal_(self.x_pos_embed, std=.02)
            trunc_normal_(self.y_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

    def forward(self, xtuple):
        x, nx, ny = xtuple
        B = x.shape[0]

        x = self.proj(x)
        nx, ny = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        assert nx == self.nx and ny == self.ny, "Fix input size!"

        if self.norm_embed:
            x = self.norm_embed(x)

        # concat cls_token
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(
                B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)

        if self.ape:
            # add position embedding
            pos_embed_2d = torch.cat([
                self.x_pos_embed.unsqueeze(2).expand(-1, -1, ny, -1),
                self.y_pos_embed.unsqueeze(1).expand(-1, nx, -1, -1),
            ], dim=-1).flatten(start_dim=1, end_dim=2)
            x = x + torch.cat([self.cls_pos_embed, pos_embed_2d], dim=1).expand(
                B, -1, -1)

        x = self.pos_drop(x)

        return x, nx, ny


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# for Performer, start
def get_module_device(module):
    return next(module.parameters()).device


def find_modules(nn_module, type):
    return [module for module in nn_module.modules() if isinstance(module, type)]

# for Performer, end


class AttnBlock(nn.Module):
    """ Meta Attn Block
    """

    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm,
                 attn_type='full', w=7, d=1, sharew=False, nglo=1,
                 only_glo=False,
                 seq_len=None, num_feats=256, share_kv=False, sw_exact=0,
                 rratio=2, rpe=False, wx=14, wy=14, mode=0, dp_rank=0):
        super().__init__()
        self.norm = norm_layer(dim)
        if attn_type == 'full':
            self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                  qk_scale=qk_scale, attn_drop=attn_drop,
                                  proj_drop=drop, rpe=rpe, wx=wx, wy=wy, nglo=nglo)
        elif attn_type == 'ls':
            # Our Long-short term attention.
            self.attn = AttentionLS(
                dim, num_heads=num_heads, qkv_bias=qkv_bias,
                qk_scale=qk_scale, attn_drop=attn_drop,
                proj_drop=drop, rpe=rpe, nglo=nglo,
                dp_rank=dp_rank, w=w
            )
        else:
            raise ValueError(
                "Not supported attention type {}".format(attn_type))
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, xtuple):
        x, nx, ny = xtuple
        x = x + self.drop_path(self.attn(self.norm(x), nx, ny))
        return x, nx, ny


class MlpBlock(nn.Module):
    """ Meta MLP Block
    """

    def __init__(self, dim, out_dim=None, mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       out_features=out_dim, act_layer=act_layer, drop=drop)
        self.shortcut = nn.Identity()
        if out_dim is not None and out_dim != dim:
            self.shortcut = nn.Sequential(nn.Linear(dim, out_dim),
                                          nn.Dropout(drop))

    def forward(self, xtuple):
        x, nx, ny = xtuple
        x = self.shortcut(x) + self.drop_path(self.mlp(self.norm(x)))
        return x, nx, ny


class MsViT(nn.Module):
    """ Multiscale Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, arch, img_size=512, in_chans=3, num_classes=1000,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 norm_embed=False, w=7, d=1, sharew=False, only_glo=False,
                 share_kv=False, attn_type='longformerhand', sw_exact=0, mode=0, **args):
        super().__init__()
        self.num_classes = num_classes
        if 'ln_eps' in args:
            ln_eps = args['ln_eps']
            self.norm_layer = partial(nn.LayerNorm, eps=ln_eps)
            logging.info("Customized LayerNorm EPS: {}".format(ln_eps))
        else:
            self.norm_layer = norm_layer
        self.drop_path_rate = drop_path_rate
        self.attn_type = attn_type

        # for performer, start
        if attn_type == "performer":
            self.auto_check_redraw = True  # TODO: make this an choice
            self.feature_redraw_interval = 1
            self.register_buffer('calls_since_last_redraw', torch.tensor(0))
        # for performer, end

        self.attn_args = dict({
            'attn_type': attn_type,
            'qkv_bias': qkv_bias,
            'qk_scale': qk_scale,
            'drop': drop_rate,
            'attn_drop': attn_drop_rate,
            'w': w,
            'd': d,
            'sharew': sharew,
            'only_glo': only_glo,
            'share_kv': share_kv,
            'sw_exact': sw_exact,
            'norm_layer': norm_layer,
            'mode': mode,
        })
        self.patch_embed_args = dict({
            'norm_layer': norm_layer,
            'norm_embed': norm_embed,
            'drop_rate': drop_rate,
        })
        self.mlp_args = dict({
            'mlp_ratio': 4.0,
            'norm_layer': norm_layer,
            'act_layer': nn.GELU,
            'drop': drop_rate,
        })

        self.Nx = img_size
        self.Ny = img_size

        def parse_arch(arch):
            layer_cfgs = []
            for layer in arch.split('_'):
                layer_cfg = {'l': 1, 'h': 3, 'd': 192, 'n': 1, 's': 1, 'g': 1,
                             'p': 2, 'f': 7, 'a': 1, 'r': 0}  # defaults. r is our low-rank attention
                for attr in layer.split(','):
                    layer_cfg[attr[0]] = int(attr[1:])
                layer_cfgs.append(layer_cfg)
            return layer_cfgs

        self.layer_cfgs = parse_arch(arch)
        self.num_layers = len(self.layer_cfgs)
        self.depth = sum([cfg['n'] for cfg in self.layer_cfgs])
        self.out_planes = self.layer_cfgs[-1]['d']
        self.Nglos = [cfg['g'] for cfg in self.layer_cfgs]
        self.avg_pool = args['avg_pool'] if 'avg_pool' in args else False
        self.dp_rank = [cfg['r'] for cfg in self.layer_cfgs]

        dprs = torch.linspace(0, drop_path_rate, self.depth).split(
            [cfg['n'] for cfg in self.layer_cfgs]
        )  # stochastic depth decay rule
        self.layer1 = self._make_layer(in_chans, self.layer_cfgs[0],
                                       dprs=dprs[0], layerid=1)
        self.layer2 = self._make_layer(self.layer_cfgs[0]['d'],
                                       self.layer_cfgs[1], dprs=dprs[1],
                                       layerid=2)
        self.layer3 = self._make_layer(self.layer_cfgs[1]['d'],
                                       self.layer_cfgs[2], dprs=dprs[2],
                                       layerid=3)
        if self.num_layers == 3:
            self.layer4 = None
        elif self.num_layers == 4:
            self.layer4 = self._make_layer(self.layer_cfgs[2]['d'],
                                           self.layer_cfgs[3], dprs=dprs[3],
                                           layerid=4)
        else:
            raise ValueError("Numer of layers {} not implemented yet!".format(self.num_layers))
        self.norm = norm_layer(self.out_planes)

        # Classifier head
        self.head = nn.Linear(self.out_planes,
                              num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _make_layer(self, in_dim, layer_cfg, dprs, layerid=0):
        layer_id, num_heads, dim, num_block, is_sparse_attn, nglo, patch_size, num_feats, ape \
            = layer_cfg['l'], layer_cfg['h'], layer_cfg['d'], layer_cfg['n'], layer_cfg['s'], layer_cfg['g'], layer_cfg['p'], layer_cfg['f'], layer_cfg['a']
        dp_rank = layer_cfg['r']

        assert layerid == layer_id, "Error in _make_layer: layerid {} does not equal to layer_id {}".format(layerid, layer_id)
        self.Nx = nx = self.Nx // patch_size
        self.Ny = ny = self.Ny // patch_size
        seq_len = nx * ny + nglo

        self.attn_args['nglo'] = nglo
        self.patch_embed_args['nglo'] = nglo
        self.attn_args['num_feats'] = num_feats  # shared for linformer and performer
        self.attn_args['rratio'] = num_feats  # srformer reuses this parameter
        self.attn_args['w'] = num_feats  # longformer reuses this parameter
        self.attn_args['dp_rank'] = dp_rank
        if is_sparse_attn == 0:
            self.attn_args['attn_type'] = 'full'

        # patch embedding
        layers = [
            PatchEmbed(patch_size, nx, ny, in_chans=in_dim, embed_dim=dim, ape=ape,
                       **self.patch_embed_args)
        ]
        for dpr in dprs:
            layers.append(AttnBlock(
                dim, num_heads, drop_path=dpr, seq_len=seq_len, rpe=not ape,
                wx=nx, wy=ny,
                **self.attn_args
            ))
            layers.append(MlpBlock(dim, drop_path=dpr, **self.mlp_args))
        return nn.Sequential(*layers)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        no_decay = {'pos_embed', 'cls_token',
                    'norm.weight', 'norm.bias',
                    'norm_embed', 'head.bias',
                    'relative_position'}
        return no_decay

    def get_classifier(self):
        return self.head

    def forward_features(self, x):
        B = x.shape[0]
        x, nx, ny = self.layer1((x, None, None))
        x = x[:, self.Nglos[0]:].transpose(-2, -1).reshape(B, -1, nx, ny)

        x, nx, ny = self.layer2((x, nx, ny))
        x = x[:, self.Nglos[1]:].transpose(-2, -1).reshape(B, -1, nx, ny)

        x, nx, ny = self.layer3((x, nx, ny))
        if self.layer4 is not None:
            x = x[:, self.Nglos[2]:].transpose(-2, -1).reshape(B, -1, nx, ny)
            x, nx, ny = self.layer4((x, nx, ny))

        x = self.norm(x)

        if self.Nglos[-1] > 0 and (not self.avg_pool):
            return x[:, 0]
        else:
            return torch.mean(x, dim=1)

    def forward(self, x):
        if self.attn_type == "performer" and self.auto_check_redraw:
            self.check_redraw_projections()
        x = self.forward_features(x)
        x = self.head(x)
        return x

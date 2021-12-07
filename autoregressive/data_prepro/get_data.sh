mkdir datasets
cd datasets

echo "- Downloading enwik8 (Character)"
if [[ ! -d 'enwik8' ]]; then
    mkdir -p enwik8
    cd enwik8
    wget --continue http://mattmahoney.net/dc/enwik8.zip
    wget https://raw.githubusercontent.com/salesforce/awd-lstm-lm/master/data/enwik8/prep_enwik8.py
    python3 prep_enwik8.py
    cd ..
fi

echo "- Downloading text8 (Character)"
if [[ ! -d 'text8' ]]; then
    mkdir -p text8
    cd text8
    wget --continue http://mattmahoney.net/dc/text8.zip
    python ../../data_prepro/prep_text8.py
    cd ..
fi
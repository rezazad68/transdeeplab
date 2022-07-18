#! /usr/bin/env bash
# run this file inside the Swin-DeepLab repository

# 1 install the dependencies
pip install -r ./requirements.txt

# 2 download the dataset
gdown --id '1BvpY0g9mKkkhdHpAX1HqDw8iTJNbFuwq' -O data.zip

# 3 unzip the data
echo 'unzipping the data ...'
unzip data.zip > /dev/null


# 4 remove the zip file
rm data.zip

# 5 move the data to the data folder
mkdir data
mv project_TransUNet/data/Synapse/* ./data/

echo 'Everything is setup successfully. run the train.py file now.'




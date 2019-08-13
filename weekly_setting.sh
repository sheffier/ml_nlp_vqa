#!/usr/bin/env bash
cd /vol/scratch && mkdir $1
cd scratch
mkdir .conda && cd .conda && mkdir envs && mkdir pkgs && cd ..
conda create -n NLP
conda activate NLP
conda update --all
conda install -c anaconda tensorflow-gpu scikit-image pyyaml jupyter
conda install -c conda-forge tqdm
conda update --all


mkdir DATASETS && cd DATASETS
git clone https://github.com/lil-lab/nlvr.git

mkdir NLVR_images && cd NLVR_images
wget "http://lil.nlp.cornell.edu/resources/NLVR2/train_img.zip" && unzip train_img.zip
wget "http://lil.nlp.cornell.edu/resources/NLVR2/dev_img.zip" && unzip dev_img.zip
wget "http://lil.nlp.cornell.edu/resources/NLVR2/test1_img.zip" && unzip test1_img.zip
rm train_img.zip
rm test1_img.zip
rm dev_img.zip
mv dev images/dev
mv test1 images/test1
cd images/train
for i in {0..99}; do mv $i/* .; done


cd ../../..
git clone --single-branch --branch two-images-handling/concat-features-horizontally https://github.com/sheffier/ml_nlp_vqa.git
cd ml_nlp_vqa/snmn/exp_nlvr
ln -s ../../../DATASETS/NLVR_images nlvr_images
ln -s ../../../DATASETS/nlvr/nlvr2/data nlvr_dataset
cd ..
./exp_nlvr/tfmodel/resnet/download_resnet_v1_152.sh
cd exp_nlvr/data
python extract_resnet152_c5_7x7.py

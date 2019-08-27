#!/usr/bin/env bash
if [[ "$#" < 2 || "$#" > 3 ]]; then
    echo "usage: $0 your-dir-name your-conda-env-name [gpu-id]"
    exit ${LINENO}
fi

my_full_dir=/vol/scratch/$1
done_marks=$my_full_dir/done
env_name=$2
gpu_id=${3:-0}

mkdir -p $done_marks || exit ${LINENO}

if [[ ! -f "$done_marks/$env_name" ]]
then
    cd $my_full_dir
    mkdir -p .conda && cd .conda && mkdir -p envs && mkdir -p pkgs || exit ${LINENO}
    conda create --yes -n $env_name
    conda activate $env_name
    conda update --yes --all
    conda install --yes -c anaconda tensorflow tensorflow-gpu scikit-image pyyaml jupyter
    conda install --yes -c conda-forge tqdm
    conda update --yes --all
    date >> "$done_marks/$env_name"
else
    conda activate $env_name
fi

cd $my_full_dir
mkdir -p DATASETS && cd DATASETS
git clone https://github.com/lil-lab/nlvr.git || ( cd nlvr && git pull )

cd $my_full_dir/DATASETS
mkdir -p NLVR_images && cd NLVR_images
if [[ ! -f "$done_marks/train" ]]
then
    wget -c "http://lil.nlp.cornell.edu/resources/NLVR2/train_img.zip" && unzip train_img.zip || exit ${LINENO}
    for i in {0..99}; do
        mv images/train/$i/* images/train
        rmdir images/train/$i
    done
    date >> "$done_marks/train"
    rm train_img.zip
fi
if [[ ! -f "$done_marks/dev" ]]
then
    wget -c "http://lil.nlp.cornell.edu/resources/NLVR2/dev_img.zip" && unzip dev_img.zip || exit ${LINENO}
    mv dev images/dev
    date >> "$done_marks/dev"
    rm dev_img.zip
fi
if [[ ! -f "$done_marks/test1" ]]
then
    wget -c "http://lil.nlp.cornell.edu/resources/NLVR2/test1_img.zip" && unzip test1_img.zip || exit ${LINENO}
    mv test1 images/test1
    date >> "$done_marks/test1"
    rm test1_img.zip
fi

cd $my_full_dir
git clone --single-branch --branch two-images-handling/concat-features-horizontally https://github.com/sheffier/ml_nlp_vqa.git || ( cd ml_nlp_vqa && git pull )

cd ml_nlp_vqa/snmn/exp_nlvr
ln -s ../../../DATASETS/NLVR_images nlvr_images
ln -s ../../../DATASETS/nlvr/nlvr2/data nlvr_dataset

if [[ ! -f "$done_marks/resnet" ]]
then
    cd $my_full_dir/ml_nlp_vqa/snmn
    bash ./exp_nlvr/tfmodel/resnet/download_resnet_v1_152.sh
    cd exp_nlvr/data
    python extract_resnet152_c5_7x7.py --gpu_id=$gpu_id
    date >> "$done_marks/resnet"
fi

cd $my_full_dir/ml_nlp_vqa/snmn
export PYTHONPATH=$(pwd):$PYTHONPATH

#!/usr/bin/env bash

usage()
{
    echo "usage: weekly_settings.sh [[[-g gpu_id ] [-e env_name]] | [-h]]"
}

DIR_NAME=""
while [ "$DIR_NAME" == "" ]
do
  echo -n "Enter base directory name > "
  read -r DIR_NAME
done

echo ""
echo "Available git branches:"
echo "======================="
echo ""
git ls-remote https://github.com/sheffier/ml_nlp_vqa.git
echo ""

GIT_BRANCH_NAME=""
while [ "$GIT_BRANCH_NAME" == "" ]
do
  echo -n "Enter git branch name > "
  read -r GIT_BRANCH_NAME
done

GPU_ID=0
CONDA_PROJ_ENV=NLP

while [ "$1" != "" ]; do
    case $1 in
        -g | --gpu_id )         shift
                                GPU_ID=$1
                                ;;
        -e | --conda_env )      shift
                                CONDA_PROJ_ENV=$1
                                ;;
        -h | --help )           usage
                                exit
                                ;;
        * )                     usage
                                exit 1
    esac
    shift
done


cd /vol/scratch || exit
source ./miniconda3/etc/profile.d/conda.sh
mkdir $DIR_NAME
cd $DIR_NAME || exit
mkdir .conda && cd .conda && mkdir envs && mkdir pkgs && cd ..
conda create -y -n $CONDA_PROJ_ENV
conda activate $CONDA_PROJ_ENV
conda update --all -y
conda install -y -c anaconda tensorflow-gpu scikit-image pyyaml jupyter
conda install -y -c conda-forge tqdm
conda update --all -y

mkdir DATASETS && cd DATASETS || exit
git clone https://github.com/lil-lab/nlvr.git

mkdir NLVR_images && cd NLVR_images || exit
wget "http://lil.nlp.cornell.edu/resources/NLVR2/train_img.zip" && unzip train_img.zip
wget "http://lil.nlp.cornell.edu/resources/NLVR2/dev_img.zip" && unzip dev_img.zip
wget "http://lil.nlp.cornell.edu/resources/NLVR2/test1_img.zip" && unzip test1_img.zip
rm train_img.zip
rm test1_img.zip
rm dev_img.zip
mv dev images/dev
mv test1 images/test1
cd images/train || exit
for i in {0..99}; do mv $i/* .; done


cd ../../../../
git clone --branch "$GIT_BRANCH_NAME" https://github.com/sheffier/ml_nlp_vqa.git
cd ml_nlp_vqa/snmn/exp_nlvr || exit
ln -s ../../../DATASETS/NLVR_images nlvr_images
ln -s ../../../DATASETS/nlvr/nlvr2/data nlvr_dataset
cd .. || exit
./exp_nlvr/tfmodel/resnet/download_resnet_v1_152.sh
cd exp_nlvr/data || exit
python extract_resnet152_c5_7x7.py --gpu_id "$GPU_ID"
python build_nlvr_imdb_r152_7x7.py
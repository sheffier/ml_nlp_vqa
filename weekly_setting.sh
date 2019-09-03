#!/usr/bin/env bash

usage () {
    echo "usage: $0 [[[-g gpu_id] [-e env_name]] | [-h]]"
    exit ${LINENO}
}

DIR_NAME=""
while [[ "$DIR_NAME" == "" ]]
do
  echo -n "Enter base directory name > "
  read -r DIR_NAME
done

printf "\nAvailable git branches:"
printf "\n=======================\n"
git ls-remote --heads https://github.com/sheffier/ml_nlp_vqa.git
echo ""

GIT_BRANCH_NAME=""
while [[ "$GIT_BRANCH_NAME" == "" ]]
do
  echo -n "Enter git branch name > "
  read -r GIT_BRANCH_NAME
done

GPU_ID=${GPU_ID:-0}
CONDA_PROJ_ENV=NLP

while [[ "$1" != "" ]]; do
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

MY_FULL_DIR=/vol/scratch/$DIR_NAME
DONE_MARKS=$MY_FULL_DIR/done

mkdir -p $DONE_MARKS || exit ${LINENO}

# executes its arguments only once; a file named `$DONE_MARKS/$1` or `$DONE_MARKS/$1-$2` is used to mark the execution
ensure_done () {
    local task_mark=$1${2:+-$2}
    if [[ -f "$DONE_MARKS/$task_mark" ]]
    then
        echo -e "\e[1;4m$task_mark\e[0m: \e[33mSkipping\e[0m (already done - \e[97m$DONE_MARKS/$task_mark\e[0m exists)"
    else
        echo -e "\e[1;4m$task_mark\e[0m: \e[34mRunning\e[0m"
        "$@"
        echo -e "\e[1;4m$task_mark\e[0m: \e[32mFinished\e[0m"
        date >> "$DONE_MARKS/$task_mark"
    fi
}

cd /vol/scratch || exit ${LINENO}
source ./miniconda3/etc/profile.d/conda.sh

create_env () {
    cd $MY_FULL_DIR
    mkdir -p .conda && cd .conda && mkdir -p envs && mkdir -p pkgs || exit ${LINENO}
    conda create --yes -n $CONDA_PROJ_ENV
    conda activate $CONDA_PROJ_ENV
    conda update --yes --all
    conda install --yes -c anaconda tensorflow tensorflow-gpu scikit-image pyyaml jupyter
    conda install --yes -c conda-forge tqdm
    pip install comet_ml
    conda update --yes --all
}
ensure_done create_env $CONDA_PROJ_ENV
conda activate $CONDA_PROJ_ENV

cd $MY_FULL_DIR
mkdir -p DATASETS && cd DATASETS
git clone https://github.com/lil-lab/nlvr.git || ( cd nlvr && git pull )

cd $MY_FULL_DIR/DATASETS
mkdir -p NLVR_images && cd NLVR_images

download_train_images () {
    wget -c "http://lil.nlp.cornell.edu/resources/NLVR2/train_img.zip" && unzip train_img.zip || exit ${LINENO}
    for i in {0..99}; do
        mv images/train/$i/* images/train
        rmdir images/train/$i
    done
    rm train_img.zip
}
download_dev_images () {
    wget -c "http://lil.nlp.cornell.edu/resources/NLVR2/dev_img.zip" && unzip dev_img.zip || exit ${LINENO}
    mv dev images/dev
    rm dev_img.zip
}
download_test1_images () {
    wget -c "http://lil.nlp.cornell.edu/resources/NLVR2/test1_img.zip" && unzip test1_img.zip || exit ${LINENO}
    mv test1 images/test1
    rm test1_img.zip
}
ensure_done download_train_images
ensure_done download_dev_images
ensure_done download_test1_images

cd $MY_FULL_DIR
git clone --branch "$GIT_BRANCH_NAME" https://github.com/sheffier/ml_nlp_vqa.git || ( cd ml_nlp_vqa ; git checkout -t -b $GIT_BRANCH_NAME ; git pull )

cd ml_nlp_vqa/snmn/exp_nlvr || exit ${LINENO}
ln -s ../../../DATASETS/NLVR_images nlvr_images
ln -s ../../../DATASETS/nlvr/nlvr2/data nlvr_dataset

extract_and_build_resnet () {
    cd $MY_FULL_DIR/ml_nlp_vqa/snmn
    bash ./exp_nlvr/tfmodel/resnet/download_resnet_v1_152.sh || exit ${LINENO}
    cd exp_nlvr/data
    python extract_resnet152_c5_7x7.py --gpu_id "$GPU_ID" || exit ${LINENO}
    python build_nlvr_imdb_r152_7x7.py || exit ${LINENO}
}
ensure_done extract_and_build_resnet

echo -e "\e[1;34mWEEKLY SETUP FINISHED.\e[0m"

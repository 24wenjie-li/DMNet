#!/usr/bin/env bash

GPUS=$1
CONFIG=$2
# PORT=${PORT:-1321}
PORT=${PORT:-$((1111 + $RANDOM % 10))}

# usage
if [ $# -lt 2 ] ;then
    echo "usage:"
    echo "./scripts/dist_train.sh [number of gpu] [path to option file]"
    exit
fi

# PYTHONPATH="$(dirname $0)/..:${PYTHONPATH}" \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    basicsr/train.py -opt $CONFIG --launcher pytorch $
    @:3}

python -m torch.distributed.launch --nproc_per_node=$GPUS basicsr/train.py -opt $CONFIG

# python -m torch.distributed.launch --nproc_per_node=1 basicsr/train.py -opt options/DLGSANet_SISR/train_try.yml --launcher pytorch
# torchrun --nproc_per_node=4 basicsr/train.py -opt options/train_restormer.yml --launcher pytorch

# pip install basicsr
# python setup.py develop -i http://mirrors.aliyun.com/pypi/simple/
# pip install -v -e .
# python -m pip install --upgrade pip
# pip install numpy==1.24.4
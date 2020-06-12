#!/usr/bin/env bash

# project directory
### CHANGE ME ###
PROJ_ROOT=/home/Deep_SLDA

export PYTHONPATH=${PROJ_ROOT}
source activate base
cd ${PROJ_ROOT}

# directory for ImageNet train and val folders
### CHANGE ME ###
IMAGENET_IMAGES_DIR=/home/data/ImageNet2012

# plastic covariance experiment
EXPT_NAME=slda_imagenet_plastic_covariance
CUDA_VISIBLE_DEVICES=0 python -u experiment.py \
  --images_dir ${IMAGENET_IMAGES_DIR} \
  --streaming_update_sigma \
  --expt_name ${EXPT_NAME} >logs/${EXPT_NAME}.log

# fixed covariance experiment
EXPT_NAME=slda_imagenet_fixed_covariance
CUDA_VISIBLE_DEVICES=0 python -u experiment.py \
  --images_dir ${IMAGENET_IMAGES_DIR} \
  --expt_name ${EXPT_NAME} >logs/${EXPT_NAME}.log

#!/bin/bash

# custom config
DATA=/work/tesi_aonori/CoOp_datasets
TRAINER=ZeroshotCLIP
DATASET=$1
CFG=$2  # rn50, rn101, vit_b32 or vit_b16
FAIRFACECLASS=$3 #select label class for FairFace

/homes/aonori/dassl_interpreter.sh /homes/aonori/Tirocinio/CoOp/train.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/CoOp/${CFG}.yaml \
--output-dir output_age1/${TRAINER}/${CFG}/${DATASET} \
--eval-only \
FAIRFACECLASS ${FAIRFACECLASS} \
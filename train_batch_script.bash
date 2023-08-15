#!/bin/bash

source /data/vision/polina/projects/wmh/dhollidt/conda/bin/activate
conda activate stratified

python /data/vision/polina/projects/wmh/dhollidt/documents/Stratified-Transformer/train.py --config /data/vision/polina/projects/wmh/dhollidt/documents/Stratified-Transformer/config/s3dis/nesf_stratified_transformer.yaml
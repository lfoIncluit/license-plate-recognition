#!/bin/bash -e

MODEL_FOLDER=./face-detection-model
MODEL=face-detection-0206

if [ ! -f "$MODEL_FOLDER/$MODEL.xml"  ]; then
    echo "Downloading Model.."
    curl https://download.01.org/opencv/2021/openvinotoolkit/2021.2/open_model_zoo/models_bin/3/face-detection-0206/FP32/face-detection-0206.xml \
     --create-dirs -o $MODEL_FOLDER/$MODEL.xml
fi

if [ ! -f "$MODEL_FOLDER/$MODEL.bin"  ]; then
    curl https://download.01.org/opencv/2021/openvinotoolkit/2021.2/open_model_zoo/models_bin/3/face-detection-0206/FP32/face-detection-0206.bin \
     --create-dirs -o $MODEL_FOLDER/$MODEL.bin
    echo "Model Downloaded"
fi
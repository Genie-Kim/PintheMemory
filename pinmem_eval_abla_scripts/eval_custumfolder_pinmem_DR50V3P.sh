#!/usr/bin/env bash
# example
# usage: ./eval_custumfolder_pinmem_DR50V3P.sh "absolute path to any image folder"
# images of qualitative results are in folder that contains snapshot pth.
cd ../
echo -e "\n\n\n\n\n\n\n\n\n\n\n\n"
set -o nounset
echo "Running inference on" ${1}

echo ${1}
python eval.py --dataset video_folder \
        --inference_mode sliding \
        --scales 1.5 \
        --split val \
        --crop_size 640 \
        --snapshot "pretrained_models/gta_synthia_idd_DR50V3P/ours/pinmem_GSI_DR50V3P.pth" \
        --dump_images \
        --exp pinmem \
        --arch network.deepv3plus.DeepR50V3PlusD \
        --memory \
        --videopath "${1}" \
        --sliding_overlap 0.5

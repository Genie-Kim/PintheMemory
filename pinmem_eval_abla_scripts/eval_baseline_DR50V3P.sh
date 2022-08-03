# example
# images of qualitative results are in folder that contains snapshot pth.
cd ../
python eval.py --dataset cityscapes \
        --inference_mode sliding \
        --scales 1.0 \
        --split val \
        --crop_size 768 \
        --snapshot "pretrained_models/GS_DR50V3P/baseline/baseline_GS_DR50V3P.pth" \
        --dump_images \
        --exp baseline \
        --arch network.deepv3plus.DeepR50V3PlusD \

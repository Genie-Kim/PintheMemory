# example
# images of qualitative results are in folder that contains snapshot pth.
cd ../
python eval.py --dataset cityscapes \
        --inference_mode sliding \
        --scales 1.0 \
        --split val \
        --crop_size 768 \
        --snapshot "pretrained_models/gta_synthia_DR50V3P/ours/pinmem_GS_DR50V3P.pth" \
        --dump_images \
        --exp pinmem \
        --arch network.deepv3plus.DeepR50V3PlusD \
        --memory

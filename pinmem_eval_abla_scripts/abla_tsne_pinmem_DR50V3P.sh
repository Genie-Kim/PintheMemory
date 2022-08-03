# example
# images of qualitative results are in folder that contains snapshot pth.
cd ../
python ablation.py --snapshot "pretrained_models/GS_DR50V3P/ours/pinmem_GS_DR50V3P.pth" \
--dataset gtav synthia cityscapes bdd100k \
--source_domain gtav synthia \
--arch network.deepv3plus.DeepR50V3PlusD \
--memory \
--tsne
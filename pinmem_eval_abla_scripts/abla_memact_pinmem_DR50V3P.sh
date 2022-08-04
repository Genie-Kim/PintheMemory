# example
# images of qualitative results are in folder that contains snapshot pth.
cd ../
python ablation.py --snapshot "pretrained_models/gta_synthia_DR50V3P/ours/pinmem_GS_DR50V3P.pth" \
--dataset cityscapes bdd100k mapillary \
--source_domain cityscapes \
--arch network.deepv3plus.DeepR50V3PlusD \
--memory \
--gumbel_off \
--mem_actmap
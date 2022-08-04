# example
# images of qualitative results are in folder that contains snapshot pth.
cd ../
python ablation.py --snapshot  "pretrained_models/gta_synthia_DR50V3P/baseline/baseline_GS_DR50V3P.pth" \
--dataset gtav synthia cityscapes bdd100k \
--source_domain gtav synthia \
--arch network.deepv3plus.DeepR50V3PlusD \
--tsne
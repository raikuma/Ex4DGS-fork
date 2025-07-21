sh scripts/train_n3dv_120k.sh coffee_martini &&
sh scripts/train_n3dv_120k.sh cook_spinach &&
sh scripts/train_n3dv_120k.sh cut_roasted_beef &&
sh scripts/train_n3dv_120k.sh flame_salmon_1 &&
sh scripts/train_n3dv_120k.sh flame_steak &&
sh scripts/train_n3dv_120k.sh sear_steak

sh metric_all.sh output/N3DV_120K/coffee_martini data/N3DV/coffee_martini/combined_motion_masks 120000 &&
sh metric_all.sh output/N3DV_120K/cook_spinach data/N3DV/cook_spinach/combined_motion_masks 120000 &&
sh metric_all.sh output/N3DV_120K/cut_roasted_beef data/N3DV/cut_roasted_beef/combined_motion_masks 120000 &&
sh metric_all.sh output/N3DV_120K/flame_salmon_1 data/N3DV/flame_salmon_1/combined_motion_masks 120000 &&
sh metric_all.sh output/N3DV_120K/flame_steak data/N3DV/flame_steak/combined_motion_masks 120000 &&
sh metric_all.sh output/N3DV_120K/sear_steak data/N3DV/sear_steak/combined_motion_masks 120000
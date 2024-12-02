task_name='METS_LinearProbing_18000'
backbone='CLIP_ResNet'
pretrain_path='/data/0shared/lijun/code/ECGFounder_CLIP/checkpoints/1st_round/resnet_encoder_18000.pth'

bash /data/0shared/lijun/code/ECGFounder_CLIP/finetune/sub_script/ptbxl/sub_ptbxl_form.sh $task_name $backbone $pretrain_path
bash /data/0shared/lijun/code/ECGFounder_CLIP/finetune/sub_script/ptbxl/sub_ptbxl_rhythm.sh $task_name $backbone $pretrain_path
bash /data/0shared/lijun/code/ECGFounder_CLIP/finetune/sub_script/ptbxl/sub_ptbxl_super_class.sh $task_name $backbone $pretrain_path
bash /data/0shared/lijun/code/ECGFounder_CLIP/finetune/sub_script/ptbxl/sub_ptbxl_sub_class.sh $task_name $backbone $pretrain_path
#!/bin/bash
batchs=1
GPU=$1
lr=0.0002
loadSize=256
fineSize=256
L1=100
size_w=640
size_h=480
model=SADC
G='SADCNet'
checkpoint='./checkpoints/'
datasetmode="shadowgttest"
dataroot='' # Need to be specify before testing
name=''  # Need to be specify before testing
NAME="${model}_G_${G}_${name}"

CMD="python ../test.py --loadSize ${loadSize} \
    --name ${NAME} \
    --dataroot  ${dataroot}\
    --checkpoints_dir ${checkpoint} \
    --size_w   $size_w   --size_h   $size_h\
    --fineSize $fineSize --model $model\
    --batch_size $batchs --keep_ratio --phase test_  --gpu_ids ${GPU} \
    --dataset_mode $datasetmode --epoch best
    --netG $G\
    $OTHER
"
echo $CMD
eval $CMD


CUDA_VISIBLE_DEVICES=0 python train.py \
 --arch vit_small \
 --m ori \
 --epochs 200 \
 --lr 1e-4 \
 --img_size 32 \
 --batch_size 256 \
 --patch_size 4 \
 --wd 0.1 \
 --num_classes 10 \
 --data_path data/ImageNet \
 --dump_path /root/result \
 --seed 1 \
 --wandb True

python train.py \
 --arch vit_small \
 --m rms \
 --epochs 200 \
 --lr 1e-4 \
 --img_size 32 \
 --batch_size 256 \
 --patch_size 4 \
 --wd 0.1 \
 --num_classes 10 \
 --data_path data/ImageNet \
 --dump_path /root/result \
 --seed 1 \
 --wandb True

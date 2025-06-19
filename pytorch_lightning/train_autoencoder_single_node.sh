export CUDA_LAUNCH_BLOCKING=1

python launch.py --config ./configs/shape-autoencoder/Dora-VAE-train.yaml \
                          --train --gpu 0,1,2,3,4,5,6,7

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# torchrun --nproc_per_node=8 launch.py --config ./configs/shape-autoencoder/Dora-VAE-train.yaml --train
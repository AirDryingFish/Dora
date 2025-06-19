import os
from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

if __name__ == "__main__":
    # 指向你的 ZeRO-2 输出目录（那个包含「checkpoint/」子文件夹的 .ckpt 目录）
    ckpt_dir = "/mnt/data/yangzengzhi/code/Dora/pytorch_lightning/outputs/Dora-VAE/test/ckpts/last-v4.ckpt"
    # 合并后想要存放的单文件路径
    out_file = "/mnt/data/yangzengzhi/code/Dora/pytorch_lightning/outputs/Dora-VAE/test/ckpts/last-v4-consolidated.ckpt"

    convert_zero_checkpoint_to_fp32_state_dict(
        checkpoint_dir=ckpt_dir,
        output_file=out_file
    )
    print(f"Consolidated checkpoint saved to {out_file}")
#  Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
# 
#  http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import math
import os
import json
from dataclasses import dataclass, field

import random
import numpy as np 
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms

from craftsman import register
from craftsman.utils.base import Updateable
from craftsman.utils.config import parse_structured
from craftsman.utils.typing import *



def random_rotation_matrix():
    """
    Create a random rotation matrix.
    """
    angle = np.random.uniform(0, 2 * np.pi)
    axis = np.random.normal(size=3)
    axis /= np.linalg.norm(axis)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    R = np.array([[cos_angle + axis[0]**2 * (1 - cos_angle),
                   axis[0]*axis[1] * (1 - cos_angle) - axis[2]*sin_angle,
                   axis[0]*axis[2] * (1 - cos_angle) + axis[1]*sin_angle],
                  [axis[1]*axis[0] * (1 - cos_angle) + axis[2]*sin_angle,
                   cos_angle + axis[1]**2 * (1 - cos_angle),
                   axis[1]*axis[2] * (1 - cos_angle) - axis[0]*sin_angle],
                  [axis[2]*axis[0] * (1 - cos_angle) - axis[1]*sin_angle,
                   axis[2]*axis[1] * (1 - cos_angle) + axis[0]*sin_angle,
                   cos_angle + axis[2]**2 * (1 - cos_angle)]])
    return R

def random_mirror_matrix():
    """
    Create a random mirror matrix.
    """
    if np.random.rand() < 0.75:
        axis = np.random.choice([0, 1, 2], size=1)[0]
        M = np.eye(3)
        M[axis, axis] = -1
    else:
        M = np.eye(3)
    return M


def apply_transformation(points, normals, transform):
    """
    Apply a transformation matrix to points and normals.
    """
    transformed_points = np.dot(points, transform.T)
    if normals is not  None:
        norms = np.linalg.norm(normals, axis=1, keepdims=True)

        epsilon = 1e-6
        norms[norms == 0] = epsilon
        norms[np.isinf(norms)] = 1
        norms[np.isnan(norms)] = 1

        normals /= norms
        
        transformed_normals = np.dot(normals, transform.T)
        norms = np.linalg.norm(transformed_normals, axis=1, keepdims=True)

        epsilon = 1e-6
        norms[norms == 0] = epsilon
        norms[np.isinf(norms)] = 1
        norms[np.isnan(norms)] = 1

        transformed_normals /= norms
    else:
        transformed_normals=None
    return transformed_points, transformed_normals

# @dataclass
# class ObjaverseDataModuleConfig:
#     root_dir: str = None
#     data_type: str = "sdf"         # sdf
    
#     load_supervision: bool = True        # whether to load supervision
#     supervision_type: str = "tsdf"       # tsdf
#     n_supervision: list[int] = field(default_factory=lambda: [21384, 10000, 10000])           # number of points in supervision

#     rotate_points: bool = False          # whether to rotate the input point cloud and the supervision
#     batch_size: int = 32
#     num_workers: int = 0

# class ObjaverseDataset(Dataset):
#     def __init__(self, cfg: Any, split: str) -> None:
#         super().__init__()
#         self.cfg: ObjaverseDataModuleConfig = cfg
#         self.split = split
#         print(f'{cfg.root_dir}/{split}.json')
#         self.uids = json.load(open(f'{cfg.root_dir}/{split}.json'))
#         print(f"Loaded {len(self.uids)} {split} uids")

#     def __len__(self):
#         if self.split =='train':
#             return len(self.uids) * 100
#         else:
#             return len(self.uids)

#     def _load_shape(self, index, mirror_matrix,rotation_matrix) -> Dict[str, Any]:
#         if self.cfg.data_type == "sdf":
#             data = np.load(f'{self.uids[index]}')
#             coarse_fps_surface = data["fps_coarse_surface"]
#             sharp_fps_surface = data['fps_sharp_surface']
#         else:
#             raise NotImplementedError(f"Data type {self.cfg.data_type} not implemented")

#         # rng = np.random.default_rng() 
#         # ind = rng.choice(5, 1, replace=False)
#         # coarse_surface = coarse_fps_surface[:,ind,:].squeeze() #（m,1,p) >(m.p)
#         coarse_surface = coarse_fps_surface[:,0,:] #（m,1,p) >(m.p)

#         nan_mask = np.isnan(coarse_surface)
#         if np.any(nan_mask):
#             print("nan exist in coarse surface")
#         coarse_surface = np.where(nan_mask, 1, coarse_surface)

#         # ind = rng.choice(5, 1, replace=False)
#         # sharp_surface = sharp_fps_surface[:,ind,:].squeeze() #（m,1,p) >(m.p)
#         sharp_surface = sharp_fps_surface[:,0,:]

#         nan_mask = np.isnan(sharp_surface)
#         if np.any(nan_mask):
#             print("nan exist in sharp surface! ")
#         sharp_surface = np.where(nan_mask, 1, sharp_surface)
        

#         if self.cfg.rotate_points and self.split=="train":
#             mirrored_points, mirrored_normals = apply_transformation(coarse_surface[:,:3], coarse_surface[:,3:], mirror_matrix)
#             surface, normal = apply_transformation(mirrored_points, mirrored_normals, rotation_matrix)
#             coarse_surface = np.concatenate([surface, normal], axis=1)
#             mirrored_points, mirrored_normals = apply_transformation(sharp_surface[:,:3], sharp_surface[:,3:], mirror_matrix)
#             surface, normal = apply_transformation(mirrored_points, mirrored_normals, rotation_matrix)
#             sharp_surface = np.concatenate([surface, normal], axis=1)

#         ret = {
#             "uid": self.uids[index],
#             "coarse_surface": coarse_surface.astype(np.float32),
#             "sharp_surface": sharp_surface.astype(np.float32),
#             "data":data
#         }

#         return ret

#     def _load_shape_supervision(self, index, mirror_matrix,rotation_matrix,data) -> Dict[str, Any]:
#         ret = {}
#         coarse_rand_points = data['rand_points'][:,:3]
#         coarse_sdfs = data['rand_points'][:,3]
#         sharp_near_points = data['sharp_near_surface'][:,:3]
#         sharp_sdfs = data['sharp_near_surface'][:,3]

#         rng = np.random.default_rng()
#         ind2 = rng.choice(sharp_near_points.shape[0], self.cfg.n_supervision[0], replace=False)
#         ind3 = rng.choice(coarse_rand_points[:400000].shape[0], self.cfg.n_supervision[1], replace=False)
#         ind4 = rng.choice(coarse_rand_points[400000:].shape[0], self.cfg.n_supervision[2], replace=False)
#         rand_points2 = sharp_near_points[ind2] 
#         rand_points3 = coarse_rand_points[:400000][ind3]
#         rand_points4 = coarse_rand_points[400000:][ind4]
#         rand_points = np.concatenate([rand_points2, rand_points3,rand_points4], axis=0) 
#         if self.cfg.rotate_points and self.split=="train":
#             mirrored_points, _ = apply_transformation(rand_points, None, mirror_matrix)
#             rand_points, _= apply_transformation(mirrored_points, None, rotation_matrix)
#         ret["rand_points"] = rand_points.astype(np.float32)
#         ret["number_sharp"] = self.cfg.n_supervision[0]
#         if self.cfg.data_type == "sdf":
#             if self.cfg.supervision_type == "occupancy":
#                 sdf2 = sharp_sdfs[ind2]
#                 sdf3 = coarse_sdfs[:400000][ind3]
#                 sdf4 = coarse_sdfs[400000:][ind4]
#                 sdfs = np.concatenate([sdf2,sdf3,sdf4], axis=0) 
#                 nan_mask = np.isnan(sdfs)
#                 if np.any(nan_mask):
#                     print("nan exist in sdfs")
#                 sdfs = np.where(nan_mask, 0, sdfs)
#                 ret["occupancies"] = np.where(sdfs.flatten() < 0, 0, 1).astype(np.float32)
#             elif self.cfg.supervision_type == "tsdf":
#                 sdf2 = sharp_sdfs[ind2]
#                 sdf3 = coarse_sdfs[:400000][ind3]
#                 sdf4 = coarse_sdfs[400000:][ind4]
#                 sdfs = np.concatenate([sdf2,sdf3,sdf4], axis=0)
#                 nan_mask = np.isnan(sdfs)
#                 if np.any(nan_mask):
#                     print("nan exist in sdfs")
#                 sdfs = np.where(nan_mask, 0, sdfs)
#                 ret["sdf"] = sdfs.flatten().astype(np.float32).clip(-0.015,0.015) / 0.015
#             else:
#                 raise NotImplementedError(f"Supervision type {self.cfg.supervision_type} not implemented")

#         return ret

#     def get_data(self, index):
#         mirror_matrix = random_mirror_matrix()
#         rotation_matrix = random_rotation_matrix()
#         ret = self._load_shape(index,mirror_matrix,rotation_matrix )
#         if self.cfg.load_supervision:
#             ret.update(self._load_shape_supervision(index,mirror_matrix,rotation_matrix,ret['data']))
#         del ret['data']
#         return ret
        
#     def __getitem__(self, index):
#         # print("getitem")
#         if self.split == 'train':
#             index %= len(self.uids)
#         try:
#             return self.get_data(index)
#         except Exception as e:
#             print(f"Error in {self.uids[index]}: {e}")
#             return self.__getitem__(np.random.randint(len(self)))


#     def collate(self, batch):
#         batch = torch.utils.data.default_collate(batch)
#         return batch

@dataclass
class ObjaverseDataModuleConfig:
    root_dir: str = None
    data_type: str = "sdf"         # sdf
    
    load_supervision: bool = True        # whether to load supervision
    supervision_type: str = "tsdf"       # tsdf
    n_supervision: list[int] = field(default_factory=lambda: [21384, 10000, 10000])           # number of points in supervision
    load_image: bool = False
    random_flip: bool = False
    random_color_jitter: bool = False
    random_rotate: bool = True
    crop_image: bool = True

    n_views: int = 1
    background_color: Tuple[int, int, int] = field(
        default_factory=lambda: (255, 255, 255)
    )


    rotate_points: bool = False          # whether to rotate the input point cloud and the supervision
    batch_size: int = 32
    num_workers: int = 0

class ObjaverseDataset(Dataset):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg: ObjaverseDataModuleConfig = cfg
        self.split = split
        print(f'{cfg.root_dir}/{split}.json')
        self.uids = json.load(open(f'{cfg.root_dir}/{split}.json'))
        print(f"Loaded {len(self.uids)} {split} uids")

        if self.cfg.random_color_jitter:
            self.color_jitter = transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2
            )

    def __len__(self):
        if self.split =='train':
            return len(self.uids) * 100
        else:
            return len(self.uids)

    def _load_shape(self, index, mirror_matrix,rotation_matrix) -> Dict[str, Any]:
        if self.cfg.data_type == "sdf":
            data = np.load(f'{self.uids[index]}')
            coarse_fps_surface = data["fps_coarse_surface"]
            sharp_fps_surface = data['fps_sharp_surface']
        else:
            raise NotImplementedError(f"Data type {self.cfg.data_type} not implemented")

        # rng = np.random.default_rng() 
        # ind = rng.choice(5, 1, replace=False)
        # coarse_surface = coarse_fps_surface[:,ind,:].squeeze() #（m,1,p) >(m.p)
        coarse_surface = coarse_fps_surface[:,0,:] #（m,1,p) >(m.p)

        nan_mask = np.isnan(coarse_surface)
        if np.any(nan_mask):
            print("nan exist in coarse surface")
        coarse_surface = np.where(nan_mask, 1, coarse_surface)

        # ind = rng.choice(5, 1, replace=False)
        # sharp_surface = sharp_fps_surface[:,ind,:].squeeze() #（m,1,p) >(m.p)
        sharp_surface = sharp_fps_surface[:,0,:]

        nan_mask = np.isnan(sharp_surface)
        if np.any(nan_mask):
            print("nan exist in sharp surface! ")
        sharp_surface = np.where(nan_mask, 1, sharp_surface)
        

        if self.cfg.rotate_points and self.split=="train":
            mirrored_points, mirrored_normals = apply_transformation(coarse_surface[:,:3], coarse_surface[:,3:], mirror_matrix)
            surface, normal = apply_transformation(mirrored_points, mirrored_normals, rotation_matrix)
            coarse_surface = np.concatenate([surface, normal], axis=1)
            mirrored_points, mirrored_normals = apply_transformation(sharp_surface[:,:3], sharp_surface[:,3:], mirror_matrix)
            surface, normal = apply_transformation(mirrored_points, mirrored_normals, rotation_matrix)
            sharp_surface = np.concatenate([surface, normal], axis=1)

        ret = {
            "uid": self.uids[index],
            "coarse_surface": coarse_surface.astype(np.float32),
            "sharp_surface": sharp_surface.astype(np.float32),
            "data":data
        }

        return ret

    def _load_shape_supervision(self, index, mirror_matrix,rotation_matrix,data) -> Dict[str, Any]:
        ret = {}
        coarse_rand_points = data['rand_points'][:,:3]
        coarse_sdfs = data['rand_points'][:,3]
        sharp_near_points = data['sharp_near_surface'][:,:3]
        sharp_sdfs = data['sharp_near_surface'][:,3]

        rng = np.random.default_rng()
        ind2 = rng.choice(sharp_near_points.shape[0], self.cfg.n_supervision[0], replace=False)
        ind3 = rng.choice(coarse_rand_points[:400000].shape[0], self.cfg.n_supervision[1], replace=False)
        ind4 = rng.choice(coarse_rand_points[400000:].shape[0], self.cfg.n_supervision[2], replace=False)
        rand_points2 = sharp_near_points[ind2] 
        rand_points3 = coarse_rand_points[:400000][ind3]
        rand_points4 = coarse_rand_points[400000:][ind4]
        rand_points = np.concatenate([rand_points2, rand_points3,rand_points4], axis=0) 
        if self.cfg.rotate_points and self.split=="train":
            mirrored_points, _ = apply_transformation(rand_points, None, mirror_matrix)
            rand_points, _= apply_transformation(mirrored_points, None, rotation_matrix)
        ret["rand_points"] = rand_points.astype(np.float32)
        ret["number_sharp"] = self.cfg.n_supervision[0]
        if self.cfg.data_type == "sdf":
            if self.cfg.supervision_type == "occupancy":
                sdf2 = sharp_sdfs[ind2]
                sdf3 = coarse_sdfs[:400000][ind3]
                sdf4 = coarse_sdfs[400000:][ind4]
                sdfs = np.concatenate([sdf2,sdf3,sdf4], axis=0) 
                nan_mask = np.isnan(sdfs)
                if np.any(nan_mask):
                    print("nan exist in sdfs")
                sdfs = np.where(nan_mask, 0, sdfs)
                ret["occupancies"] = np.where(sdfs.flatten() < 0, 0, 1).astype(np.float32)
            elif self.cfg.supervision_type == "tsdf":
                sdf2 = sharp_sdfs[ind2]
                sdf3 = coarse_sdfs[:400000][ind3]
                sdf4 = coarse_sdfs[400000:][ind4]
                sdfs = np.concatenate([sdf2,sdf3,sdf4], axis=0)
                nan_mask = np.isnan(sdfs)
                if np.any(nan_mask):
                    print("nan exist in sdfs")
                sdfs = np.where(nan_mask, 0, sdfs)
                ret["sdf"] = sdfs.flatten().astype(np.float32).clip(-0.015,0.015) / 0.015
            else:
                raise NotImplementedError(f"Supervision type {self.cfg.supervision_type} not implemented")

        return ret
    
    def _load_image(self, index: int) -> Dict[str, Any]:
        def _process_img(image, background_color=(255, 255, 255), foreground_ratio=0.9):
            alpha = image.getchannel("A")
            background = Image.new("RGBA", image.size, (*background_color, 255))
            image = Image.alpha_composite(background, image)
            image = image.crop(alpha.getbbox())

            new_size = tuple(int(dim * foreground_ratio) for dim in image.size)
            resized_image = image.resize(new_size)
            padded_image = Image.new("RGBA", image.size, (*background_color, 255))
            paste_position = (
                (image.width - resized_image.width) // 2,
                (image.height - resized_image.height) // 2,
            )
            padded_image.paste(resized_image, paste_position)

            # Expand image to 1:1
            max_dim = max(padded_image.size)
            image = Image.new("RGBA", (max_dim, max_dim), (*background_color, 255))
            paste_position = (
                (max_dim - padded_image.width) // 2,
                (max_dim - padded_image.height) // 2,
            )
            image.paste(padded_image, paste_position)
            # image = image.resize((512, 512))
            image = image.resize((518, 518))
            return image.convert("RGB"), alpha

        ret = {}
        # if self.cfg.image_type == "rgb" or self.cfg.image_type == "normal":
        assert (
            self.cfg.n_views == 1
        ), "Only single view is supported for single image"
        sel_idx = random.choice(self.cfg.idx)
        ret["sel_image_idx"] = sel_idx
        # if self.cfg.image_type == "rgb":

        # @todo: to be modified
        sha = os.path.splitext(os.path.basename(self.uids[index]))[0]   # e.g. "01f79c9741984ef2b855e936f35bf34d"
        img_path = os.path.join(
            "/mnt/data/yangzengzhi/data/renders_cond",  # or use self.cfg.root_dir
            sha,
            f"{sel_idx:03d}.png"                       # e.g. "018.png"
        )
        # img_path = ( 
        #     "/mnt/data/yangzengzhi/data/renders_cond/"
        #     + "/".join(self.uids[index].split("/")[-2:])
        #     + f"/{'{:04d}'.format(sel_idx)}_rgb.{self.cfg.image_file_type}"
        # )
        # elif self.cfg.image_type == "normal":
        #     img_path = (
        #         f"{self.cfg.root_dir}/images/"
        #         + "/".join(self.uids[index].split("/")[-2:])
        #         + f"/{'{:04d}'.format(sel_idx)}_normal.{self.cfg.image_file_type}"
        #     )
        image = Image.open(img_path).copy()

        # add random color jitter
        if self.cfg.random_color_jitter:
            rgb = self.color_jitter(image.convert("RGB"))
            image = Image.merge("RGBA", (*rgb.split(), image.getchannel("A")))

        # add random rotation
        if self.cfg.random_rotate:
            image = self.rotate(image)

        # add crop
        if self.cfg.crop_image:
            background_color = (
                torch.randint(0, 256, (3,))
                if self.cfg.background_color is None
                else torch.as_tensor(self.cfg.background_color)
            )
            image, alpha = _process_img(
                image, background_color, self.cfg.foreground_ratio
            )
        else:
            alpha = image.getchannel("A")
            background = Image.new("RGBA", image.size, background_color)
            image = Image.alpha_composite(background, image).convert("RGB")

        ret["image"] = torch.from_numpy(np.array(image) / 255.0)
        ret["mask"] = torch.from_numpy(np.array(alpha) / 255.0).unsqueeze(0)
        # else:
        #     raise NotImplementedError(
        #         f"Image type {self.cfg.image_type} not implemented"
        #     )

        return ret

    def get_data(self, index):
        mirror_matrix = random_mirror_matrix()
        rotation_matrix = random_rotation_matrix()

        flip = np.random.rand() < 0.5 if self.cfg.random_flip else False

        ret = self._load_shape(index,mirror_matrix,rotation_matrix)

        # 沿着x轴翻转（左右翻转）
        if flip:  # random flip the input point cloud and the supervision
            for key in ret.keys():
                if key in ["surface", "sharp_surface"]:  # N x (xyz + normal)
                    ret[key][:, 0] = -ret[key][:, 0]
                    ret[key][:, 3] = -ret[key][:, 3]
                elif key in ["rand_points"]:
                    ret[key][:, 0] = -ret[key][:, 0]
        
        if self.cfg.load_supervision:
            ret.update(self._load_shape_supervision(index,mirror_matrix,rotation_matrix,ret['data']))

        if self.cfg.load_image:
            ret.update(self._load_image(index))
            if flip:  # random flip the input image
                for key in ret.keys():
                    if key in ["image"]:  # random flip the input image
                        ret[key] = torch.flip(ret[key], [2])
                    if key in ["mask"]:  # random flip the input image
                        ret[key] = torch.flip(ret[key], [2])

        del ret['data']
        return ret
        
    def __getitem__(self, index):
        # print("getitem")
        if self.split == 'train':
            index %= len(self.uids)
        try:
            return self.get_data(index)
        except Exception as e:
            print(f"Error in {self.uids[index]}: {e}")
            return self.__getitem__(np.random.randint(len(self)))


    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        return batch


@register("objaverse-datamodule")
class ObjaverseDataModule(pl.LightningDataModule):
    cfg: ObjaverseDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(ObjaverseDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        # print("setup")
        if stage in [None, "fit"]:
            self.train_dataset = ObjaverseDataset(self.cfg, "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = ObjaverseDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = ObjaverseDataset(self.cfg, "test")

    def prepare_data(self):
        # print("prepare_data")
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None, num_workers=0) -> DataLoader:
        # print("general_loader")
        return DataLoader(
            dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers
        )

    def train_dataloader(self) -> DataLoader:
        # print("train_dataloader")
        return self.general_loader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            collate_fn=self.train_dataset.collate,
            num_workers=self.cfg.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        # print("val_dataloader")
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1)

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1)
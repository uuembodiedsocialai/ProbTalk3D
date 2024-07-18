from pathlib import Path
import numpy as np
import pickle
from natsort import os_sorted

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

tmp_vert = Path("../datasets/templates_mead_vert.pkl")
with open(tmp_vert, 'rb') as f:
    template = pickle.load(f, encoding='latin1')    # [5023, 3]

tmp_zero = Path("../datasets/flame_zero_pose.npy")
template_zero = np.load(tmp_zero)                   # (5023, 3)


# remove the shape in the ground truth data
def remove_shape_gt(files):
    for file in files:
        motion = np.load(file)
        motion[:, :, :300] = 0
        save_path = Path(file.parent) / f"{file.stem}_zero_pose_shape.npy"
        np.save(save_path, motion)
        print("Save file at:", save_path)


def remove_shape_vert(files):
    for idx, file in enumerate(files):
        motion = np.load(file)                                  # (T, 15069)
        motion = np.reshape(motion, (-1, 15069 // 3, 3))        # (T, 5023, 3)

        file_split = file.stem.split("_")
        template_np = template[file_split[0]].detach().numpy()  # (5023, 3)

        displacement = motion - template_np
        new_motion = template_zero + displacement               # (T, 5023, 3)

        save_path = file.parent / f"{file.stem}(noshape).npy"
        np.save(save_path, new_motion)
        print("Save file at:", save_path)


if __name__ == "__main__":
    '''remove the shape in the ground truth data'''
    # npy_foler = Path(f"{parent_dir}/results/generation/vqvae_pred/stage_2/0.2")
    # files = list(npy_foler.glob("*.npy"))
    # remove_shape_gt(files)

    '''remove the shape from the pred vert data that has shape info (FaceDiffuser)'''
    # npy_foler = Path(f"{parent_dir}/results/generation/vqvae_pred/stage_2/0.2")
    # files = list(npy_foler.glob("*.npy"))
    # files = os_sorted(files)
    # remove_shape_vert(files)

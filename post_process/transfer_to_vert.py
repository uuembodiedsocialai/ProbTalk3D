from pathlib import Path
import numpy as np
import torch
import trimesh

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from deps.flame.flame_pytorch import FLAME, get_config


# template compute: transfer zero pose to vertices
def zero_transfer_to_vert(dir, ref_path):
    dir = Path(dir)
    config = get_config(batch_size=1)       # set the batch size
    flamelayer = FLAME(config)
    input_shape = torch.zeros(1, 300)
    input_exp = torch.zeros(1, 100)
    input_pose = torch.zeros(1, 6)

    vertice, landmark = flamelayer(input_shape, input_exp, input_pose)
    vertice = vertice.squeeze(0)            # (5023, 3)
    vertice = vertice.detach().cpu().numpy()
    save_path = Path(dir)/f"flame_zero_pose.npy"
    np.save(save_path, vertice)
    print(f"Save npy file at: {save_path}")

    template_all = trimesh.load(ref_path)
    template_all.vertices = vertice
    save_path_obj = Path(dir)/f"flame_zero_pose.obj"
    template_all.export(save_path_obj)
    print(f"Save obj file at: {save_path_obj}, you can view it!")


# transfer pred FLAME param to vert
def transfer_to_vert(dir):
    dir = Path(dir)
    seq = np.load(dir)
    seq = np.squeeze(seq)           # (T, 403)
    m_tensor = torch.tensor(seq, dtype=torch.float32)

    config = get_config(batch_size=m_tensor.shape[0])           # set the batch size
    flamelayer = FLAME(config)
    input_shape = m_tensor[:, :300]
    input_exp = m_tensor[:, 300:400]
    input_global_pose = torch.zeros([m_tensor.shape[0], 3])     # no global head rotation
    input_pose = torch.cat((input_global_pose, m_tensor[:, 400:]), dim=1)

    vertice, landmark = flamelayer(input_shape, input_exp, input_pose)
    vertice = vertice.squeeze(0)    # (T, 5023, 3)
    save_path = Path(dir).parent/f"{dir.stem}(vert).npy"
    np.save(save_path, vertice.detach().cpu().numpy())
    print(f"Save file at: {save_path}")


if __name__ == "__main__":
    '''compute zero pose vert template'''
    # motion_dir = Path(f"{parent_dir}/datasets")
    # ref_path = Path(f"{parent_dir}/datasets/flame_sample.ply")
    # zero_transfer_to_vert(motion_dir, ref_path)

    '''transfer pred FLAME param to vert '''
    npy_foler = Path(f"{parent_dir}/results/generation/vqvae_pred/stage_2/0.2")
    files = list(npy_foler.glob("*.npy"))
    for file in files:
        transfer_to_vert(file)




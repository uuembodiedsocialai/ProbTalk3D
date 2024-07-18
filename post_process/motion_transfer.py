from pathlib import Path
import numpy as np
import torch
import pickle

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from deps.flame.flame_pytorch import FLAME, get_config

# Execute in blender: export a new shape from blender flame addon.
def blender_export_new():
    import bpy
    import numpy as np
    # Select the object containing the vertices
    obj = bpy.context.active_object
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated_object = obj.evaluated_get(depsgraph)
    evaluated_mesh = evaluated_object.data
    vertices_array = np.array([v.co for v in evaluated_mesh.vertices])

    # Save vertices coordinates to a .npy file
    np.save('C://new_face.npy', vertices_array)


# Transfer motion from one identity to another (FLAME parameters -> vertices)
def motion_transfer_param(motion_dir, tmp_dir):
    input = np.load(motion_dir)
    input = torch.tensor(input, dtype=torch.float32)    # [1, T, 403]

    # transfer  to vertices
    config = get_config(batch_size=input.shape[1])      # set the batch size to T
    flamelayer = FLAME(config)

    input[:, :, :300] *= 0.
    input_shape = input[:, :, :300]
    input_exp = input[:, :, 300:400]
    input_global_pose = torch.zeros([input.shape[0], input.shape[1], 3], device=input.device)
    input_pose = torch.cat((input_global_pose, input[:, :, 400:]), dim=2)
    vertice, landmark = flamelayer(
        input_shape.squeeze(0), input_exp.squeeze(0), input_pose.squeeze(0)
    )
    # print("Vertice shape", vertice.shape)       # [T, 5023, 3]

    # transfer zero pose to vertices
    input_shape = torch.zeros(1, 300)
    input_exp = torch.zeros(1, 100)
    input_pose = torch.zeros(1, 6)
    config = get_config(batch_size=1)           # set the batch size to 1
    flamelayer = FLAME(config)
    vertice_zero, landmark = flamelayer(
        input_shape, input_exp, input_pose
    )                                           # [1, 5023, 3]

    # get the displacement
    displace = vertice.detach().cpu().numpy() - vertice_zero.detach().cpu().numpy()     # [T, 5023, 3]

    # load new face template
    temp = np.load(tmp_dir)                     # (5023, 3)
    rotation_matrix = np.array([[1, 0, 0],
                                [0, 0, 1],
                                [0, -1, 0]])
    temp = np.dot(temp, rotation_matrix.T)      # Perform the rotation by matrix multiplication
    new_template_motion = temp + displace       # (48, 5023, 3)

    save_path = f"{motion_dir.parent}/{motion_dir.stem}({tmp_dir.stem}).npy"
    np.save(save_path, new_template_motion)
    print(f"Save file at: {save_path}")


# Transfer motion from one identity to another (vertices -> vertices): for vert preds has shape info (FaceDiffuser)
def motion_transfer_vert(motion_dir, tmp_dir):
    motion = np.load(motion_dir)                            # (T, 15069)
    motion = np.reshape(motion, (-1, 15069 // 3, 3))        # (T, 5023, 3)

    template_path = Path("../datasets/templates_mead_vert.pkl")
    with open(template_path, 'rb') as f:
        template = pickle.load(f, encoding='latin1')        # [5023, 3]

    # this only works for the dataset predictions named like: M005_029_6_2, change it id needed!!!
    file_split = motion_dir.stem.split("_")
    template_np = template[file_split[0]].detach().numpy()  # (5023, 3)
    print("Original face id:", file_split[0])
    displacement = motion - template_np

    temp_new = np.load(tmp_dir)  # (5023, 3)
    rotation_matrix = np.array([[1, 0, 0],
                                [0, 0, 1],
                                [0, -1, 0]])

    temp_new = np.dot(temp_new, rotation_matrix.T)          # Perform the rotation by matrix multiplication
    new_template_motion = temp_new + displacement           # (48, 5023, 3)
    save_path = f"{motion_dir.parent}/{motion_dir.stem}({tmp_dir.stem}).npy"
    np.save(save_path, new_template_motion)
    print(f"Save file at: {save_path}")


if __name__ == "__main__":
    ''' transfer the pred FLAME parameters to vertices with a new face template'''
    # motion_dir = Path(f"{parent_dir}/results/generation/vqvae_pred/stage_2/0.2")
    # tmp_dir = Path("anim_transfer/newface1.npy")
    # files = list(motion_dir.glob("*.npy"))
    # for file in files:
    #     motion_transfer_param(file, tmp_dir)

    ''' transfer the pred vertices to a new face template'''
    # motion_dir = Path(f"{parent_dir}/results/generation/vqvae_pred/stage_2/0.2")
    # tmp_dir = Path("anim_transfer/newface2.npy")
    # files = list(motion_dir.glob("*.npy"))
    # for file in files:
    #     motion_transfer_vert(file, tmp_dir)



import numpy as np
import torch
from pathlib import Path
from rich.progress import Progress
import pickle

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from deps.flame.flame_pytorch import FLAME, get_config

IDs = ['M003', 'M005', 'M007', 'M009', 'M011', 'M012', 'M013', 'M019',
       'M022', 'M023', 'M024', 'M025', 'M026', 'M027', 'M028', 'M029',
       'M030', 'M031', 'W009', 'W011', 'W014', 'W015', 'W016', 'W018',
       'W019', 'W021', 'W023', 'W024', 'W025', 'W026', 'W028', 'W029']
IDs.extend(['M032', 'M033', 'M034', 'M035', 'W033', 'W035', 'W036'])
IDs.extend(['M037', 'M039', 'M040', 'M041', 'M042', 'W037', 'W038', 'W040'])


# compute vertex template
def vertex_template():
    savepath = Path('../datasets')
    print(f"Template computing script. The result will be stored in: {savepath}")
    motionpath = Path('../datasets/mead/param')
    temp_dict = {}

    # flame parameters setting
    config = get_config(batch_size=1)  # set the batch size
    flamelayer = FLAME(config)
    input_exp = torch.zeros(1, 100)
    input_pose = torch.zeros(1, 6)

    with Progress(transient=False) as progress:
        task = progress.add_task("Computing", total=len(IDs))
        for id in IDs:
            progress.update(task, description=f"Computing {id}..")
            motion_dir = list(motionpath.glob(f"{id}*.npy"))

            all_motion_id = []
            for dir in motion_dir:
                m_npy = np.load(dir)
                m_tensor = torch.tensor(m_npy)
                m_tensor[:, :, 300:] *= 0.
                all_motion_id.append(m_tensor)

            all_motion_id = torch.cat(all_motion_id, dim=1)
            average_shape = torch.mean(all_motion_id, dim=1)
            average_shape = average_shape.unsqueeze(0)

            input_shape = average_shape[:, :, :300]
            vertice, landmark = flamelayer(input_shape.squeeze(0), input_exp, input_pose)
            vertice = vertice.squeeze(0)

            temp_dict[id] = vertice
            print(f"Template computed for {id}")
            progress.update(task, advance=1)

        savepath = savepath / "templates_mead_vert.pkl"
        with open(savepath, 'wb') as f:
            pickle.dump(temp_dict, f)
        print("Computing done.")


# convert parameters to vertices
def convert_param_to_vert():
    savepath = Path('../datasets/mead/vertex')
    savepath.mkdir(exist_ok=True, parents=True)
    print(f"Converting script. The outputs will be stored in: {savepath}")

    motionpath = Path('../datasets/mead/param')
    with Progress(transient=False) as progress:
        task = progress.add_task("Converting", total=len(IDs))
        for id in IDs:
            progress.update(task, description=f"Converting {id}..")
            motion_dir = list(motionpath.glob(f"{id}*.npy"))

            for dir in motion_dir:
                m_npy = np.load(dir)
                m_tensor = torch.tensor(m_npy)
                m_tensor = m_tensor.squeeze(0)

                # flame parameters setting
                config = get_config(batch_size=m_tensor.shape[0])  # set the batch size
                flamelayer = FLAME(config)
                input_shape = m_tensor[:, :300]
                input_exp = m_tensor[:, 300:400]
                input_neck_pose = torch.zeros([m_tensor.shape[0], 3])
                input_pose = torch.cat((input_neck_pose, m_tensor[:, 400:]), dim=1)

                vertice, landmark = flamelayer(input_shape, input_exp, input_pose)
                vertice = vertice.squeeze(0)

                savepath_temp = savepath / f"{dir.stem}.npy"
                np.save(savepath_temp, vertice)
            progress.update(task, advance=1)
            print(f"Converting completed for {id}")

        print(f"Converting done.")


if __name__ == '__main__':
    # vertex_template()
    convert_param_to_vert()


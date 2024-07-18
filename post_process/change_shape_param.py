from pathlib import Path
import numpy as np
import pickle
from natsort import os_sorted

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

# shape templates, contains 47 face shapes computed from the 3DMEAD
tmp = Path("../datasets/templates_mead_param.pkl")
with open(tmp, 'rb') as f:
    template = pickle.load(f, encoding='latin1')


# This function change face shape to the FLAME parameter motion data
def get_shape(files, ids):
    for idx, file in enumerate(files):
        motion = np.load(file)
        print("Processing file:", file.stem)
        print("Add face shape id:", ids[idx])
        shape = template[ids[idx]]
        shape = shape.repeat(1, motion.shape[1], 1)         # (1, T, 300)
        shape = shape.detach().numpy()

        motion = motion[:, :, 300:]
        motion = np.concatenate((shape, motion), axis=2)    # (1, T, 403)

        save_path = file.parent / f"{file.stem}({ids[idx]}).npy"

        np.save(save_path, motion)
        save_path = save_path.resolve()
        print("Saving file at: ", save_path)


if __name__ == "__main__":
    npy_foler = Path(f"{parent_dir}/results/generation/vqvae_pred/stage_2/0.2")
    files = list(npy_foler.glob("*.npy"))
    files = os_sorted(files)
    keys = [file.stem for file in files]

    # set the shape id
    shapes = ["M003", "M011", "M012", "M028", "W015", "W018", "W019", "W026"]
    get_shape(files, shapes)

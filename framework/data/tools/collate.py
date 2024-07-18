from typing import List, Dict
from sklearn.preprocessing import minmax_scale
import torch
from torch import Tensor

# For training motion prediction
def collate_motion_and_audio(lst_elements: List) -> Dict:
    batch = {
        # Collate motion
        "motion": [x["motion"] for x in lst_elements],
        # Collate audio
        "audio": [x["audio"] for x in lst_elements]}

    # add other keys, such as keyid
    otherkeys = [x for x in lst_elements[0].keys() if x not in batch]
    for key in otherkeys:
        batch[key] = [x[key] for x in lst_elements]

    return batch


# For training motion prior
def collate_motion(lst_elements: List) -> Dict:
    batch = {"motion": [x["motion"] for x in lst_elements]}
    # add other keys, such as keyid
    otherkeys = [x for x in lst_elements[0].keys() if x not in batch]
    for key in otherkeys:
        batch[key] = [x[key] for x in lst_elements]

    return batch


def audio_normalize(data):
    normalized_data = minmax_scale(data, feature_range=(-1, 1))
    return normalized_data


def lengths_to_mask(lengths: List[int], device: torch.device) -> Tensor:
    lengths = torch.tensor(lengths, device=device)
    max_len = max(lengths)
    mask = torch.arange(max_len, device=device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask

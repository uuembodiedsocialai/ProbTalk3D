from typing import List
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


def load_checkpoint(model, ckpt_path, *, eval_mode, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    logger.info("Model weights restored.")

    if eval_mode:
        model.eval()
        logger.info("Model in eval mode.")


def detach_to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def create_one_hot(keyids: List[str], IDs_list: List, IDs_labels: List[List], one_hot_dim: List):
    style_batch = []
    for keyid in keyids:
        keyid = keyid.split('_')
        identity_vector = IDs_labels[IDs_list.index(keyid[0])]
        emotion_idx = int(keyid[2])
        emotion_vector = torch.eye(one_hot_dim[1])[emotion_idx]
        intensity_idx = int(keyid[3])
        intensity_vector = torch.eye(one_hot_dim[2])[intensity_idx]
        style_vector = torch.cat([identity_vector, emotion_vector, intensity_vector], dim=0)
        style_batch.append(style_vector)
    style_batch = torch.stack(style_batch, dim=0)
    return style_batch


def resample_input(audio_embed, motion_embed, ifps, ofps):
    if ifps % ofps == 0:
        factor = -1 * (-ifps // ofps)
        if audio_embed.shape[1] % 2 != 0:
            audio_embed = audio_embed[:, :audio_embed.shape[1] - 1, :]

        if audio_embed.shape[1] > motion_embed.shape[1] * 2:
            audio_embed = audio_embed[:, :motion_embed.shape[1] * 2, :]
        elif audio_embed.shape[1] < motion_embed.shape[1] * 2:
            motion_embed = motion_embed[:, :audio_embed.shape[1] // 2, :]
    else:
        factor = -1 * (-ifps // ofps)
        audio_embed_len = motion_embed.shape[1] * factor
        audio_embed = audio_embed.permute(0, 2, 1)
        audio_embed = F.interpolate(audio_embed, size=audio_embed_len, align_corners=True, mode='linear')
        audio_embed = audio_embed.permute(0, 2, 1)

    batch_size = motion_embed.shape[0]
    audio_embed = torch.reshape(audio_embed,
                                (batch_size, audio_embed.shape[1] // factor, audio_embed.shape[2] * factor))
    return audio_embed, motion_embed
import pytorch_lightning as pl
import logging
from pathlib import Path
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from rich.progress import Progress
import pickle
import torch
import numpy as np

import framework.launch.prepare  # noqa
from framework.model.utils.tools import load_checkpoint, detach_to_numpy
from framework.data.tools.collate import collate_motion_and_audio
from deps.flame.flame_pytorch import FLAME
from generation import cfg_mean_nsamples_resolution, get_path_vae, get_path_vqvae


logger = logging.getLogger(__name__)

emo_dict = {
       "0": "neutral",  # only have one intensity level
       "1": "happy",
       "2": "sad",
       "3": "surprised",
       "4": "fear",
       "5": "disgusted",
       "6": "angry",
       "7": "contempt"
   }

int_dict = {
       "0": "low",
       "1": "medium",
       "2": "high",
   }

# load scalers
with open("datasets/scaler_exp.pkl", 'rb') as f:
    scaler_exp = pickle.load(f)
with open("datasets/scaler_jaw.pkl", 'rb') as f:
    scaler_jaw = pickle.load(f)


@hydra.main(version_base=None, config_path="configs", config_name="evaluation")
def _sample(cfg: DictConfig):
    sample(cfg)


def sample(newcfg: DictConfig):
    # Load previous configs
    prevcfg = OmegaConf.load(Path(newcfg.folder) / ".hydra/config.yaml")
    # Merge configs to overload them
    cfg = OmegaConf.merge(prevcfg, newcfg)

    onesample = cfg_mean_nsamples_resolution(cfg)

    logger.info("Sample script. The outputs will be stored in:")
    folder_name = cfg.folder.split("/")[-1]
    output_dir = Path(cfg.path.code_dir) / f"results/evaluation/{cfg.experiment}/{folder_name}"
    path = None
    if hasattr(cfg.model, 'vae_pred') and cfg.model.vae_pred:
        path = get_path_vae(output_dir, onesample, cfg.mean, cfg.fact)
    if hasattr(cfg.model, 'vqvae_pred') and cfg.model.vqvae_pred:
        if not cfg.sample:
            path = get_path_vqvae(output_dir, onesample, "none", cfg.k)
        else:
            path = get_path_vqvae(output_dir, onesample, cfg.temperature, cfg.k)
    if path is None:
        raise ValueError("No model specified in the config file.")
    else:
        path.mkdir(exist_ok=True, parents=True)
        logger.info(f"{path}")

    # save config to check
    OmegaConf.save(cfg, output_dir / "merged_config.yaml")
    pl.seed_everything(cfg.seed)

    logger.info("Loading data module")
    data_module = instantiate(cfg.data)
    logger.info(f"Data module '{cfg.data.data_name}' loaded")

    logger.info("Loading model")
    last_ckpt_path = cfg.last_ckpt_path
    logger.info(last_ckpt_path)
    model = instantiate(cfg.model,
                        nfeats=data_module.nfeats,
                        split_path=data_module.split_path,
                        one_hot_dim=data_module.one_hot_dim,
                        resumed_training=False,
                        logger_name="none",
                        _recursive_=False)
    logger.info(f"Model '{cfg.model.modelname}' loaded")
    # move model to cuda
    if cfg.device is None:
        device_index = cfg.trainer.devices[0]
    else:
        device_index = cfg.device
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        if device_index < num_devices:
            model.to(f"cuda:{device_index}")
        else:
            model.to(f"cuda:0")
    print("device checking:", model.device)

    # load ckpt
    load_checkpoint(model, last_ckpt_path, eval_mode=True, device=model.device)
    if hasattr(cfg.model, 'vae_pred') and cfg.model.vae_pred:
        model.motion_prior.sample_mean = cfg.mean
        model.motion_prior.fact = cfg.fact
    if hasattr(cfg.model, 'vqvae_pred') and cfg.model.vqvae_pred:
        model.temperature = cfg.temperature
        model.k = cfg.k

    # load test data
    dataset = getattr(data_module, f"{cfg.split}_dataset")

    # remove printing for changing the seed
    logging.getLogger('pytorch_lightning.utilities.seed').setLevel(logging.WARNING)

    # load templates for evaluation
    with open(Path(cfg.region_path)/"lve.txt")as f:
        maps = f.read().split(",")
        mouth_map = [int(i) for i in maps]
    with open(Path(cfg.region_path)/"fdd.txt") as f:
        maps = f.read().split(",")
        upper_map = [int(i) for i in maps]
    templates_ref = np.load(cfg.reference_path)     # (5023, 3) zero pose

    seq_count = 0
    frame_count = 0
    vertices_all_gt = []        # mve, lve: gt
    vertices_all_pred = []      # mve, lve: first prediction
    mee_all = []                # mean value
    ce_all = []                 # closest value
    motion_std_difference = []  # fdd: first prediction
    diversity = 0               # 2 subsets
    with torch.no_grad():
        with Progress(transient=True) as progress:
            task = progress.add_task("Sampling", total=len(dataset.keyids))
            for idx, keyid in enumerate(dataset.keyids):
                progress.update(task, description=f"Sampling {keyid}..")

                # load gt data
                motion_gt = np.load(Path(cfg.data.motion_path) / f"{keyid}.npy")    # (1, T, 403)
                # save gt to check (a small part)
                keyid_split = keyid.split("_")
                emo = emo_dict[str(keyid_split[2])]     # retrieve emotion
                ints = int_dict[str(keyid_split[3])]    # retrieve intensity
                gt_path = path / "param" / f"{keyid}_{emo}_{ints}_gt.npy"
                gt_path_vert = path / "vert" / f"{keyid}_{emo}_{ints}_gt.npy"

                if (idx + 1) % 100 == 0:
                    logger.info(f"Saving param: {gt_path.stem}")
                    gt_path.parent.mkdir(exist_ok=True, parents=True)
                    np.save(gt_path, motion_gt)

                # sample test data
                test_data = dataset.load_keyid(keyid)
                batch = collate_motion_and_audio([test_data])
                
                ce_lve_set = []     # save 10 lve values
                motion_set = []     # save 10 samples
                vertices_npy_gt = None
                flamelayer = None
                for i in range(cfg.number_of_samples):
                    motion_pred = model(batch.copy(), sample=cfg.sample, generation=False)
                    if motion_pred.shape[1] < motion_gt.shape[1] and i == 0:
                        motion_gt = motion_gt[:, :motion_pred.shape[1], :]      # (1, T, 403)
                    assert motion_gt.shape[1] == motion_pred.shape[1], "Length mismatch"

                    # denormalization
                    shape = np.zeros((motion_gt.shape[1], 300))     # (T, 300) use zero shape
                    if cfg.number_of_samples > 1:
                        pred_path = path/"param"/f"{keyid}_{emo}_{ints}_{i}.npy"
                        pred_path_vert = path/"vert"/f"{keyid}_{emo}_{ints}_{i}.npy"
                    else:
                        pred_path = path/"param"/f"{keyid}_{emo}_{ints}_one.npy"
                        pred_path_vert = path/"vert"/f"{keyid}_{emo}_{ints}_one.npy"

                    inverse_exp_pred, inverse_jaw_pred = denormalization(shape=shape,
                                                                         exp_jaw=detach_to_numpy(motion_pred.squeeze(0)),
                                                                         save_path=pred_path,
                                                                         idx=idx,
                                                                         i=i)
                    if i == 0:  # set the flame layer and compute gt vertices for once
                        # initialize flame layer
                        cfg.flame.batch_size = motion_gt.shape[1]
                        flamelayer = FLAME(cfg.flame).to(model.device)
                        # adjust input shape
                        exp_gt = np.squeeze(motion_gt[:, :, 300:350])           # (T, 50) use first 50 exp
                        exp_suffix = np.zeros((motion_gt.shape[1], 50))         # (T, 50)
                        exp_gt = np.concatenate((exp_gt, exp_suffix), axis=1)   # (T, 100)
                        jaw_gt = np.squeeze(motion_gt[:, :, 400:])              # (T, 3)
                        # transfer gt params to vertex (T, 5023, 3)
                        vertices_npy_gt = transfer_to_vert(flamelayer=flamelayer,
                                                           shape=shape,
                                                           exp=exp_gt,
                                                           jaw=jaw_gt,
                                                           save_path=gt_path_vert,
                                                           device=model.device,
                                                           idx=idx,
                                                           i=i)
                    assert flamelayer is not None and vertices_npy_gt is not None, "No GT information"

                    # transfer pred params to vertex (T, 5023, 3)
                    vertices_npy_pred = transfer_to_vert(flamelayer=flamelayer,
                                                         shape=shape,
                                                         exp=inverse_exp_pred,
                                                         jaw=inverse_jaw_pred,
                                                         save_path=pred_path_vert,
                                                         device=model.device,
                                                         idx=idx,
                                                         i=i)
                    """CE: compute lve for each samples"""
                    ce_lve = lve_compute(vertices_gt=list(vertices_npy_gt),
                                         vertices_pred=list(vertices_npy_pred),
                                         mouth_map=mouth_map)
                    ce_lve_set.append(ce_lve)                   # (T,)
                    
                    # save 10 samples
                    motion_set.append(vertices_npy_pred)        # (T, 5023, 3)
                    torch.cuda.empty_cache()

                """MVE, LVE: save prediction of all audio samples"""
                vertices_all_gt.extend(list(vertices_npy_gt))   # length T of items (5023, 3)
                vertices_all_pred.extend(list(motion_set[0]))   # use the first sample

                """MEE: mean over 10 samples"""
                motion_set_stack = np.stack(motion_set, axis=0)
                vertices_npy_pred_mean = np.mean(motion_set_stack, axis=0)      # (T, 5023, 3)
                mee_lve = lve_compute(vertices_gt=list(vertices_npy_gt), 
                                      vertices_pred=list(vertices_npy_pred_mean), 
                                      mouth_map=mouth_map)
                mee_all.extend(list(mee_lve))

                """CE: closest lve in 10 samples"""
                smallest_lve = None
                smallest_lve_value = float('inf')               # start with an infinitely large value
                for lve_of_one_seq in ce_lve_set:
                    lve_value = np.sum(lve_of_one_seq)
                    if lve_value < smallest_lve_value:
                        smallest_lve_value = lve_value
                        smallest_lve = lve_of_one_seq           # (T,)
                assert smallest_lve is not None, "No smallest distance found"
                ce_all.extend(list(smallest_lve))

                # count sequence, frame numbers
                frame_count += motion_gt.shape[1]
                seq_count += 1

                """FDD computation: use the first sample"""
                subject_template = templates_ref.reshape(1, 5023, 3)
                upper_std_gt = seq_std_compute(motion=vertices_npy_gt - subject_template,
                                               map=upper_map)
                upper_std_pred = seq_std_compute(motion=motion_set[0] - subject_template,
                                                 map=upper_map)
                motion_std_difference.append(upper_std_gt - upper_std_pred)

                """Diversity computation"""
                np.random.shuffle(motion_set)           # list of (T, 5023, 3) number=10
                subset1 = motion_set[:5]
                subset2 = motion_set[5:]
                motion_diversity = 0
                for sample1, sample2 in zip(subset1, subset2):
                    motion_diversity += np.linalg.norm(sample1 - sample2, axis=2).mean(axis=1).mean()
                if len(subset1) == 5 and len(subset2) == 5:
                    motion_diversity /= len(subset1)
                    diversity += motion_diversity
                else:
                    raise ValueError("Subset length mismatch 5")

                print(f"Done sampling: {keyid} ")       # condition on {cdt_id}")
                torch.cuda.empty_cache()
                progress.update(task, advance=1)

    # logging.disable(logging.NOTSET)
    logger.info('Total sequence number: {}'.format(seq_count))
    logger.info('Total frame number: {}'.format(frame_count))
    if seq_count == 0:
        logger.info("No sequences were processed. Unable to compute metrics.")
    else:
        """MVE computation"""
        vertices_all_gt = np.array(vertices_all_gt)     # (frame_cunt, 5023, 3)
        vertices_all_pred = np.array(vertices_all_pred)
        vertices_dis = np.linalg.norm(vertices_all_gt - vertices_all_pred, axis=2)
        logger.info('MVE: {:.4e}'.format(np.mean(vertices_dis)))

        """LVE computation"""
        L2_dis_mouth_max = lve_compute(vertices_gt=vertices_all_gt,
                                       vertices_pred=vertices_all_pred,
                                       mouth_map=mouth_map)
        logger.info('LVE: {:.4e}'.format(np.mean(L2_dis_mouth_max)))

        """MEE computation"""
        logger.info('MEE: {:.4e}'.format(np.mean(mee_all)))

        """CE computation"""
        logger.info('CE: {:.4e}'.format(np.mean(ce_all)))

        """FDD computation"""
        logger.info('FDD: {:.4e}'.format(sum(motion_std_difference) / len(motion_std_difference)))

        """Divertiy computation"""
        logger.info('Diversity: {:.4e}'.format(diversity / seq_count))

        logger.info(f"All the sampling are done. You can find them here:\n{path}")


def denormalization(shape, exp_jaw, save_path, idx, i):
    exp_suffix = np.zeros((exp_jaw.shape[0], 50))   # (T, 50)

    inverse_exp = scaler_exp.inverse_transform(exp_jaw[:, :50])
    inverse_jaw = scaler_jaw.inverse_transform(exp_jaw[:, 50:])
    inverse_exp = np.concatenate((inverse_exp, exp_suffix), axis=1)         # (T, 100)
    inverse_exp_jaw = np.concatenate((inverse_exp, inverse_jaw), axis=1)    # (T, 103)
    seq = np.concatenate((shape, inverse_exp_jaw), axis=1)                  # (T, 403)
    seq = np.expand_dims(seq, axis=0)               # (1, T, 403)
    # save params (a small part)
    if (idx+1) % 100 == 0:  # save every 100 keyids for checking
        if (i+1) % 3 == 1:  # save 3 samples in 10
            logger.info(f"Saving param: {save_path.stem}")
            save_path.parent.mkdir(exist_ok=True, parents=True)
            np.save(save_path, seq)
    return inverse_exp, inverse_jaw


def transfer_to_vert(flamelayer, shape, exp, jaw, save_path, device, idx, i):
    input_shape = torch.tensor(shape).to(device)    # [T, 300]
    input_exp = torch.tensor(exp).to(device)        # [T, 100]
    input_global_pose = torch.zeros(jaw.shape[0], 3).to(device)    # [T, 3]
    input_jaw_pose = torch.tensor(jaw).to(device)   # [T, 3]
    input_pose = torch.cat((input_global_pose, input_jaw_pose), dim=1)
    # transfer params to vertices [T, 5023, 3]
    vertices, _ = flamelayer(input_shape.float(), input_exp.float(), input_pose.float())
    vertices_npy = detach_to_numpy(vertices)
    # save vertices (a small part)
    if (idx+1) % 100 == 0:  # save every 100 keyids for checking
        if (i+1) % 3 == 1:  # save 3 samples in 10
            logger.info(f"Saving vert: {save_path.stem}")
            save_path.parent.mkdir(exist_ok=True, parents=True)
            np.save(save_path, vertices_npy)
    return vertices_npy


def lve_compute(vertices_gt, vertices_pred, mouth_map):
    # L2_dis_mouth_max: (428, T, 3), 428 vertex indices
    vertices_gt = np.array(vertices_gt)
    vertices_pred = np.array(vertices_pred)
    L2_dis_mouth_max = np.array([np.square(vertices_gt[:, v, :] - vertices_pred[:, v, :]) for v in mouth_map])
    L2_dis_mouth_max = np.transpose(L2_dis_mouth_max, (1, 0, 2))    # (T, 428, 3)
    L2_dis_mouth_max = np.sum(L2_dis_mouth_max, axis=2)
    L2_dis_mouth_max = np.max(L2_dis_mouth_max, axis=1)
    return L2_dis_mouth_max


def seq_std_compute(motion, map):
    # map: 1501 vertex indices
    # motion[:, v, :]: (T, 3)
    L2_dis = np.array([np.square(motion[:, v, :]) for v in map])    # (1501, T, 3)
    L2_dis = np.transpose(L2_dis, (1, 0, 2))                        # (T, 1501, 3)
    L2_dis = np.sum(L2_dis, axis=2)
    L2_dis = np.std(L2_dis, axis=0)
    std = np.mean(L2_dis)
    return std


if __name__ == '__main__':
    _sample()

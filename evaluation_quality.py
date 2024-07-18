import logging
from pathlib import Path
import hydra
from omegaconf import DictConfig
import numpy as np
import pyrender
import pymeshlab as pmlab
import cv2
import trimesh
import gc
import matplotlib.pyplot as plt

import framework.launch.prepare  # noqa


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="evaluation_quality")
def _evaluation(cfg: DictConfig):
    heatmap(cfg)
    # animation_curve(cfg)


def heatmap(cfg: DictConfig):
    all_dir = list(Path(cfg.pred_path).glob("*.npy"))
    pred_dir = [file for file in all_dir if "gt" not in file.name]

    output_path = Path(cfg.pred_path) / "heatmap"
    output_path.mkdir(exist_ok=True, parents=True)

    template_all = trimesh.load(cfg.reference_path)
    template_all = template_all.vertices
    template_all = template_all.astype(np.float64)

    for i, directory in enumerate(pred_dir):
        name_split = directory.stem.split("_")
        print("Processing:", directory.stem)
        """render predict result"""
        seq_frames = np.load(directory)   # (T, 5023, 3)
        motion_vectors = np.linalg.norm(seq_frames[1:, :, :] - seq_frames[:-1, :, :], axis=2)
        mean_motion = np.mean(motion_vectors, axis=0)
        print("MEAN prediction max", max(mean_motion))
        # render mean
        render_heatmap(reference_path=cfg.reference_path, template_all=template_all,
                           motion_vec=mean_motion, output_path=str(output_path/f"{directory.stem}_mean.png"))

        # render std
        std_motion = np.std(motion_vectors, axis=0)
        print("STD prediction max", max(std_motion))
        render_heatmap(reference_path=cfg.reference_path, template_all=template_all,
                           motion_vec=std_motion, output_path=str(output_path / f"{directory.stem}_std.png"))

        """render ground truth, this gt has been processed to have zero shape"""
        name_split[-1] = "gt"
        filename = "_".join(name_split)
        gt = directory.parent / f"{filename}.npy"
        seq_frames_2 = np.load(gt)
        motion_vectors_gt = np.linalg.norm(seq_frames_2[1:, :, :] - seq_frames_2[:-1, :, :], axis=2)
        mean_motion_gt = np.mean(motion_vectors_gt, axis=0)
        print("MEAN gt max", max(mean_motion_gt))
        # render mean
        render_heatmap(reference_path=cfg.reference_path, template_all=template_all,
                           motion_vec=mean_motion_gt, output_path=str(output_path / f"{filename}_mean.png"))
        # render std
        std_motion_gt = np.std(motion_vectors_gt, axis=0)
        print("STD gt max", max(std_motion_gt))
        render_heatmap(reference_path=cfg.reference_path, template_all=template_all,
                           motion_vec=std_motion_gt, output_path=str(output_path / f"{filename}_std.png"))
    print("Save file at:", output_path)


def render_heatmap(reference_path, template_all, motion_vec, output_path):
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 20.0, aspectRatio=1.414)
    camera_pose = np.array([[1.0, 0, 0.0, 0.00],
                            [0.0, 1.0, 0.0, 0.00],
                            [0.0, 0.0, 1.0, 3.0],
                            [0.0, 0.0, 0.0, 1.0]])
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=10.0)

    r = pyrender.OffscreenRenderer(960, 760)

    ms = pmlab.MeshSet()
    ms.load_new_mesh(reference_path)
    template_mesh = ms.current_mesh()
    mesh_render = pmlab.Mesh(template_all, template_mesh.face_matrix(),
                             template_mesh.vertex_normal_matrix(), v_scalar_array=motion_vec)
    ms.add_mesh(mesh_render)
    ms.apply_filter('compute_color_from_scalar_using_transfer_function_per_vertex',
                    minqualityval=motion_vec.min(),
                    maxqualityval=motion_vec.max(), tfslist=2, brightness=1)
    ms.save_current_mesh(f'{Path(output_path).parent}/tmp.obj', save_vertex_color=True)

    ref_mesh = trimesh.load_mesh(f'{Path(output_path).parent}/tmp.obj')
    ref_mesh.visual.vertx_colors = np.random.uniform(size=ref_mesh.vertices.shape)
    py_mesh = pyrender.Mesh.from_trimesh(ref_mesh)

    scene = pyrender.Scene()
    node = pyrender.Node(
        mesh=py_mesh,
        translation=[0, 0, 0]
    )
    scene.add_node(node)
    scene.add(cam, pose=camera_pose)
    scene.add(light, pose=camera_pose)
    color, _ = r.render(scene)
    cv2.imwrite(output_path, color)

    del ref_mesh
    gc.collect()


def animation_curve(cfg: DictConfig):
    pred_dir = list(Path(cfg.pred_path_multi).glob("*.npy"))
    gt_path = Path(cfg.motionpath)
    output_path = Path(cfg.pred_path_multi) / "curve"
    output_path.mkdir(exist_ok=True, parents=True)

    gt_motion = None
    gt_name = None
    logger.info("EXP parameters")
    for i, directory in enumerate(pred_dir):
        pred_motion = np.load(directory)
        print("Load pred", directory)

        if i == 0:
            name = directory.stem.split("_")
            gt_name = "_".join(name[:-3]) + ".npy"
            gt_motion = np.load(gt_path / gt_name)
            print("Load gt", gt_path / gt_name)
            gt_motion = gt_motion[:, :pred_motion.shape[1], :]
            # plot expression
            gt_motion_exp = gt_motion[:, :, 300:350]
            gt_feature_exp = np.mean(gt_motion_exp, axis=-1)
            plt.plot(range(gt_motion_exp.shape[1]), gt_feature_exp[0], label='GT')
        assert gt_motion is not None, "Ground truth motion is not loaded"
        pred_motion = pred_motion[:, :gt_motion.shape[1], :]
        assert gt_motion.shape == pred_motion.shape, "Shape mismatch"

        # plot
        pred_motion_exp = pred_motion[:, :, 300:350]
        pred_feature_exp = np.mean(pred_motion_exp, axis=-1)
        plt.plot(range(pred_motion_exp.shape[1]), pred_feature_exp[0], label=f'pred{i+1}')

    # add labels and legend
    plt.xlabel('Frame Index')
    plt.ylabel('Mean Value')
    plt.title('Expression Parameters')
    plt.legend()
    plt.savefig(output_path / f'{gt_name[:-4]}_exp.png')
    plt.close()

    logger.info("JAW parameters")
    for i, directory in enumerate(pred_dir):
        pred_motion = np.load(directory)
        print("Load pred", directory)
        if i == 0:
            # plot jaw
            gt_motion_jaw = gt_motion[:, :, 400:]
            gt_feature_jaw = np.mean(gt_motion_jaw, axis=-1)
            plt.plot(range(gt_motion_jaw.shape[1]), gt_feature_jaw[0], label='GT')
        pred_motion = pred_motion[:, :gt_motion.shape[1], :]
        assert gt_motion.shape == pred_motion.shape, "Shape mismatch"
        # plot
        pred_motion_jaw = pred_motion[:, :, 400:]
        pred_feature_jaw = np.mean(pred_motion_jaw, axis=-1)
        plt.plot(range(pred_motion_jaw.shape[1]), pred_feature_jaw[0], label=f'pred{i+1}')

    # add labels and legend
    plt.xlabel('Frame Index')
    plt.ylabel('Mean Value')
    plt.title('Jaw Parameters')
    plt.legend()
    plt.savefig(output_path / f'{gt_name[:-4]}_jaw.png')
    plt.close()


if __name__ == '__main__':
    _evaluation()

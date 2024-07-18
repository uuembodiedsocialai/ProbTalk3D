from pathlib import Path
import trimesh
import numpy as np
import cv2
import os
import shutil
import ffmpeg
import gc
import pyrender
import logging
from omegaconf import DictConfig
import hydra
import framework.launch.prepare  # noqa
import platform
if platform.system() == "Linux":
    os.environ["PYOPENGL_PLATFORM"] = "egl"
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="render_vert")
def _render(cfg: DictConfig):
    return render(cfg)


def render(cfg: DictConfig) -> None:
    seqs = os.listdir(cfg.result_folder)

    fps = 25
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 20, aspectRatio=1.414)
    camera_pose = np.array([[1.0, 0, 0.0, 0.00],
                            [0.0, 1.0, 0.0, 0.00],
                            [0.0, 0.0, 1.0, 3.0],
                            [0.0, 0.0, 0.0, 1.0]])

    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=10.0)
    r = pyrender.OffscreenRenderer(960, 760)

    if not os.path.exists(Path(cfg.video_woA_folder)):
        os.makedirs(Path(cfg.video_woA_folder))
    if not os.path.exists(Path(cfg.video_wA_folder)):
        os.makedirs(Path(cfg.video_wA_folder))
    if not os.path.exists(Path(cfg.frames_folder)):
        os.makedirs(Path(cfg.frames_folder))

    for seq in seqs:
        if seq.endswith('.npy'):
            video_woA_path = Path(cfg.video_woA_folder)/f"{seq.split('.')[0]}.mp4"
            video_wA_path = Path(cfg.video_wA_folder)/f"{seq.split('.')[0]}.mp4"
            video = cv2.VideoWriter(str(video_woA_path), fourcc, fps, (960, 760))
            motion_path = Path(cfg.result_folder)/seq

            wav_name = motion_path.stem.split("_")
            wav_name = "_".join(wav_name[:-3])
            wav_path = Path(cfg.audio_folder) / f"{wav_name}.wav"

            if not os.path.isfile(wav_path):
                wav_path = Path(cfg.audio_folder) / f"{motion_path.stem}.wav"   # GT
                if not os.path.isfile(wav_path):
                    logger.error(f"No file named: {wav_path}")
            else:
                print("Loading audio:", wav_path)
            if not os.path.isfile(motion_path):
                logger.error(f"No file named: {motion_path}")
            else:
                print("Loading result:", motion_path)

            ref_mesh = trimesh.load_mesh(cfg.subject_template_path, process=False)

            seq = np.load(motion_path)
            seq = np.squeeze(seq)
            seq = np.reshape(seq,(-1,15069//3,3))

            for f in range(seq.shape[0]):
                ref_mesh.vertices = seq[f, :, :]
                py_mesh = pyrender.Mesh.from_trimesh(ref_mesh)

                scene = pyrender.Scene()
                scene.add(py_mesh)
                scene.add(cam, pose=camera_pose)
                scene.add(light, pose=camera_pose)
                color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)

                output_frame = Path(cfg.frames_folder)/f"{motion_path.stem}_frame{str(f)}.jpg"
                cv2.imwrite(str(output_frame), color)
                frame = cv2.imread(str(output_frame))
                video.write(frame)
            video.release()

            input_video = ffmpeg.input(str(video_woA_path))
            input_audio = ffmpeg.input(str(wav_path))
            audio_exists = os.path.exists(str(wav_path))
            if audio_exists:
                ffmpeg.concat(input_video, input_audio, v=1, a=1).output(str(video_wA_path)).run(overwrite_output=True)
            else:
                ffmpeg.concat(input_video, v=1, a=0).output(str(video_wA_path)).run(overwrite_output=True)

            del video, seq, ref_mesh
            gc.collect()

    # Cleanup
    cv2.destroyAllWindows()
    if os.path.exists(Path(cfg.frames_folder)):
        shutil.rmtree(Path(cfg.frames_folder))
    if os.path.exists(Path(cfg.video_woA_folder)):
        shutil.rmtree(Path(cfg.video_woA_folder))
    logger.info(f"Video saved at: {cfg.video_wA_folder}")


if __name__ == "__main__":
    _render()

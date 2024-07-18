import os, subprocess
import shlex
import logging
import hydra
from omegaconf import DictConfig
from pathlib import Path
import framework.launch.prepare  # noqa
from rich.progress import Progress, BarColumn, TimeRemainingColumn


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="render_param")
def _render(cfg: DictConfig):
    return render(cfg)


def render_video(cfg: DictConfig, wav_path: Path, filename: str) -> None:
    image_path = Path(cfg.resultMotion_folder) / filename
    os.makedirs(image_path, exist_ok=True)
    image_temp = image_path / "%d.png"

    output_path = cfg.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_path = Path(output_path) / f"{filename}.mp4"

    cmd = '{} -t 64 -b {} -P {} -- "{}" "{}" '.format(
        Path(cfg.blender_path),
        Path(cfg.blend_path),
        Path(cfg.renderScript_path),
        str(Path(cfg.resultMotion_folder).resolve()) + '\\',
        filename)
    cmd = cmd.replace('\\', '/')
    cmd = shlex.split(cmd)
    p = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    if cfg.progress_bar:
        progress = Progress(
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "•",
            TimeRemainingColumn(),
        )
        with progress:
            task = progress.add_task(f"Rendering...")
            # Read the subprocess output line
            for line in iter(p.stdout.readline, b''):
                # Update the progress bar for each line read
                progress.update(task, advance=1)
    p.wait()

    if os.path.isfile(wav_path):
        cmd = 'ffmpeg -r 25 -i "{}" -i "{}" -pix_fmt yuv420p -s 1280x720 "{}" -y -hide_banner -loglevel error'\
            .format(image_temp, wav_path, output_path)
    else:
        cmd = 'ffmpeg -r 25 -i "{}" -pix_fmt yuv420p -s 1280x720 "{}" -y -hide_banner -loglevel error' \
            .format(image_temp, output_path)
    subprocess.call(cmd, shell=True)

    cmd = 'rd /s /q "{}"'.format(image_path)
    subprocess.call(cmd, shell=True)


def render(cfg: DictConfig) -> None:
    resultMotion_folder = Path(cfg.resultMotion_folder)
    audio_folder = Path(cfg.audio_folder)
    sequences = os.listdir(resultMotion_folder)

    for sequence in sequences:
        if sequence.endswith('.npy'):
            filename = Path(sequence).stem
            motion_path = resultMotion_folder/sequence

            wav_name = filename.split("_")
            wav_name = "_".join(wav_name[:-3])
            wav_path = Path(audio_folder) / f"{wav_name}.wav"

            if not os.path.isfile(wav_path):
                wav_path = Path(audio_folder) / f"{filename}.wav"   # GT
                if not os.path.isfile(wav_path):
                    logger.error(f"No file named: {wav_path}")
            else:
                print("Loading audio:", wav_path)
            if not os.path.isfile(motion_path):
                logger.error(f"No file named: {motion_path}")
            else:
                print("Loading result:", motion_path)

            render_video(cfg, wav_path, filename)

    logger.info(f"Video saved at: {cfg.output_path}")


if __name__ == "__main__":
    _render()

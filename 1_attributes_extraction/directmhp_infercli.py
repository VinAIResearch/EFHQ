from pathlib import Path
from typing import List

import torch
import typer
import yaml
from models.experimental import attempt_load
from tqdm.rich import tqdm
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device
from zipdatasetcustom import ImageZipDatasetMHP


app = typer.Typer()


@app.command()
def main(
    zippath: Path = typer.Argument(..., help="path: zippath, zipfilelist, zipbasedir"),
    output_basepath: Path = typer.Argument(..., help="output basepath"),
    datapath: Path = typer.Option("data/agora_coco.yaml", help="Path to data.yaml"),
    device_id: int = typer.Option(0, help="Device ID to use"),
    weights: Path = typer.Option("weights/agora_m_1280_e300_t40_lw010_best.pt", help="Path to weights"),
    imgsz: int = typer.Option(1280, help="imgsz"),
    scales: List[float] = typer.Option([0.25, 0.5, 0.75, 1, 1.25, 1.5, 3.0], help="img scale"),
    conf_thres: float = typer.Option(0.7, help="conf thresh"),
    iou_thres: float = typer.Option(0.45, help="iou thresh"),
):
    with open(datapath) as f:
        data = yaml.safe_load(f)  # load data dict

    device = select_device(device_id, batch_size=1)
    print(f"Using device: {device}")

    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    paths = []
    if zippath.is_file():
        if zippath.suffix.lower == ".zip":
            paths = [zippath]
        elif zippath.suffix == ".txt":
            with open(zippath) as f:
                lines = f.readlines()
                paths = list(map(lambda x: Path(x.strip()), lines))
    elif zippath.is_dir():
        paths = list(zippath.rglob("*.zip"))
    if len(paths) == 1:
        pathiter = iter(paths)
    else:
        pathiter = tqdm(paths)

    for zippath in pathiter:
        parent_path = output_basepath / zippath.stem
        parent_path.mkdir(exist_ok=True, parents=True)
        dataset_wrapper = ImageZipDatasetMHP(zippath, imgsz, stride, auto=True)
        with dataset_wrapper.dataset() as dataset:
            dataset_iter = iter(dataset)
            if len(paths) == 1:
                idxiter = tqdm(range(len(dataset)))
            else:
                idxiter = range(len(dataset))
            for index in idxiter:
                (single_path, img, im0, _) = next(dataset_iter)
                single_path = Path(single_path)
                final_outpath = parent_path / single_path.with_suffix(".txt").name
                if final_outpath.exists():
                    continue

                img = torch.from_numpy(img).to(device)
                img = img / 255.0  # 0 - 255 to 0.0 - 1.0
                if len(img.shape) == 3:
                    img = img[None]  # expand for batch dim

                out_ori = model(img, augment=True, scales=scales)[0]
                out = non_max_suppression(out_ori, conf_thres, iou_thres, num_angles=data["num_angles"])

                # predictions (Array[N, 9]), x1, y1, x2, y2, conf, class, pitch, yaw, roll
                bboxes = scale_coords(img.shape[2:], out[0][:, :4], im0.shape[:2]).cpu().numpy()  # native-space pred
                pitchs_yaws_rolls = out[0][:, 6:].cpu().numpy()  # N*3

                with open(final_outpath, "w") as f:
                    for i, [x1, y1, x2, y2] in enumerate(bboxes):
                        pitch = (pitchs_yaws_rolls[i][0] - 0.5) * 180
                        yaw = (pitchs_yaws_rolls[i][1] - 0.5) * 360
                        roll = (pitchs_yaws_rolls[i][2] - 0.5) * 180
                        f.write(f"{yaw:.3f} {pitch:.3f} {roll:.3f} {x1:.3f} {y1:.3f} {x2:.3f} {y2:.3f}\n")
        dataset_wrapper.zipfile.close()


if __name__ == "__main__":
    app()

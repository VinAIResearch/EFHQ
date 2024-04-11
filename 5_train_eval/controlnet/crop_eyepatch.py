from functools import partial
from multiprocessing import Pool
from pathlib import Path

import mediapipe as mp
import numpy as np
import typer
from path_utils import get_filelist_and_cache
from PIL import Image
from tqdm import tqdm


eye_indices = list(mp.solutions.face_mesh_connections.FACEMESH_LEFT_EYE) + list(
    mp.solutions.face_mesh_connections.FACEMESH_RIGHT_EYE
)
eye_indices = np.unique(np.array(eye_indices).reshape(-1)).tolist()


def select_best_landmarks(face_landmarks):
    if face_landmarks.shape[0] == 1:
        return face_landmarks[0]
    else:
        best_idx = 0
        biggest_area = -1
        for i, landmarks in enumerate(face_landmarks):
            # Extract the x and y coordinates of all 68 facial landmarks
            x_coordinates = [landmarks[i][0] for i in range(face_landmarks.shape[1])]
            y_coordinates = [landmarks[i][1] for i in range(face_landmarks.shape[1])]

            # Calculate the minimum and maximum x and y coordinates
            min_x = min(x_coordinates)
            max_x = max(x_coordinates)
            min_y = min(y_coordinates)
            max_y = max(y_coordinates)
            w = max_x - min_x
            h = max_y - min_y
            area = w * h
            if area > biggest_area:
                best_idx = i
                biggest_area = area
        return face_landmarks[best_idx]


app = typer.Typer()


def f(image_path, padding, indir, outdir, lmkdir):
    fpath = outdir / image_path.relative_to(indir)
    if fpath.exists():
        return

    lmk_path = lmkdir / image_path.relative_to(indir).with_suffix(".npy")
    if not lmk_path.exists():
        print(f"{lmk_path} does not exist")
        return
    multiple_landmarks = np.load(lmk_path)
    landmarks = select_best_landmarks(multiple_landmarks)
    assert len(landmarks.shape) == 2, f"Landmark invalid shape: {landmarks.shape}"
    x_coordinates = [landmarks[i][0] for i in eye_indices]
    y_coordinates = [landmarks[i][1] for i in eye_indices]

    image = Image.open(image_path.as_posix())
    # Calculate the minimum and maximum x and y coordinates
    min_x = min(x_coordinates)
    max_x = max(x_coordinates)
    min_y = min(y_coordinates)
    max_y = max(y_coordinates)
    min_x = max(0, min_x - padding)
    max_x = min(image.size[0], max_x + padding)
    min_y = max(0, min_y - padding)
    max_y = min(image.size[1], max_y + padding)
    eye_patch = image.crop((min_x, min_y, max_x, max_y))
    fpath.parent.mkdir(parents=True, exist_ok=True)
    eye_patch.save(fpath.as_posix())


@app.command()
def main(
    indir: Path = typer.Argument(..., help="Input dir"),
    lmkdir: Path = typer.Argument(..., help="mediapipe landmark dir"),
    outdir: Path = typer.Argument(..., help="Output dir"),
    padding: int = typer.Option(5, help="Padding"),
    nprocs: int = typer.Option(16, help="Nprocs"),
):
    image_paths = get_filelist_and_cache(indir, "*.png")
    image_paths = list(filter(lambda x: "condition" not in x.name, image_paths))

    with Pool(nprocs) as p:
        list(
            tqdm(
                p.imap(
                    partial(
                        f,
                        outdir=outdir,
                        indir=indir,
                        lmkdir=lmkdir,
                        padding=padding,
                    ),
                    image_paths,
                ),
                total=len(image_paths),
            )
        )

    pass


if __name__ == "__main__":
    app()

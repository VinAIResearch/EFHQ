from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import typer
from mediapipeface import draw_pupils_nparray, face_connection_spec, generate_annotation, iris_landmark_spec
from path_utils import get_filelist_and_cache
from PIL import Image
from tqdm import tqdm


mp_drawing = mp.solutions.drawing_utils
app = typer.Typer()


def select_best_landmarks(face_landmarks):
    if face_landmarks.shape[0] == 1:
        return face_landmarks[0]
    else:
        best_idx = 0
        biggest_area = -1
        for i, landmarks in enumerate(face_landmarks):
            # Extract the x and y coordinates of all 68 facial landmarks
            x_coordinates = [landmarks[i][0] for i in range(landmarks.shape[0])]
            y_coordinates = [landmarks[i][1] for i in range(landmarks.shape[0])]

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


def f(
    img_path,
    image_dir,
    output_dir,
    min_image_size: int = 384,
    max_image_size: int = 32766,
    min_face_size_pixels: int = 64,
):
    outpath = output_dir / img_path.relative_to(image_dir).with_suffix(".npy")
    if outpath.exists():
        return
    try:
        img = Image.open(img_path.as_posix()).convert("RGB")
    except OSError:
        return
    img_width = img.size[0]
    img_height = img.size[1]
    img_size = min(img.size[0], img.size[1])
    if img_size < min_image_size or max(img_width, img_height) > max_image_size:
        return

    # We re-initialize the detector every time because it has a habit of triggering weird race conditions.
    landmarks = generate_annotation(
        img,
        max_faces=1,
        min_face_size_pixels=min_face_size_pixels,
        return_landmarks_only=True,
    )
    output = []
    nfaces = len(landmarks)
    if nfaces == 0:
        return
    for facelmk in landmarks:
        single_output = []
        for lmk in facelmk.landmark:
            x = int(lmk.x * img_width)
            y = int(lmk.y * img_height)
            single_output.append([x, y])
        single_output = np.array(single_output).reshape(-1, 2)

        output.append(single_output)
    output = np.array(output).reshape(nfaces, -1, 2)

    outpath = output_dir / img_path.relative_to(image_dir).with_suffix(".npy")
    outpath.parent.mkdir(exist_ok=True, parents=True)
    np.save(outpath.as_posix(), output)


@app.command()
def extract(
    image_dir: Path = typer.Argument(..., help="image dir", dir_okay=True, exists=True),
    output_dir: Path = typer.Argument(..., help="output dir", dir_okay=True),
    nprocs: int = typer.Option(16, help="Nprocs"),
):
    img_paths = list(get_filelist_and_cache(image_dir, "*.[jp][pn]g"))
    with Pool(nprocs) as p:
        list(
            tqdm(
                p.imap(
                    partial(
                        f,
                        image_dir=image_dir,
                        output_dir=output_dir,
                    ),
                    img_paths,
                ),
                total=len(img_paths),
            )
        )


@app.command()
def draw_lmk_dlib(
    img_path: Path = typer.Argument(..., help="image_path"),
    npy_path: Path = typer.Argument(..., help="image_path"),
    out_path: Path = typer.Argument(..., help="image_path"),
    idx: str = typer.Option(None, help="image_path"),
):
    def parse_idx(idx: str):
        split = idx.split(",")
        ids = list(map(lambda x: int(x.strip()), split))
        return ids

    ids = []
    if idx is not None:
        ids = parse_idx(idx)
        assert len(ids) > 0
    img = cv2.imread(img_path.as_posix())
    d = np.load(npy_path.as_posix())
    for i in range(d.shape[0]):
        lmks = d[i]
        for j, lmk in enumerate(lmks):
            if idx is not None and j not in ids:
                continue
            cv2.circle(img, (lmk[0], lmk[1]), 1, (0, 0, 255), -1)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(out_path.as_posix(), img)


def atomic_draw_lmk_mediapipe(npy_path, npy_dir, out_dir):
    out_path = out_dir / npy_path.relative_to(npy_dir).with_suffix(".png")
    if out_path.exists():
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    multiple_landmarks = np.load(npy_path.as_posix())
    landmarks = select_best_landmarks(multiple_landmarks)
    empty = np.zeros((512, 512, 3), dtype=np.uint8)
    mp_drawing.draw_landmarks_npyarray(
        empty,
        landmarks,
        connections=face_connection_spec.keys(),
        landmark_drawing_spec=None,
        connection_drawing_spec=face_connection_spec,
    )
    draw_pupils_nparray(empty, landmarks, iris_landmark_spec, 2)
    cv2.imwrite(out_path.as_posix(), empty)


@app.command()
def draw_lmk_mediapipe(
    npy_dir: Path = typer.Argument(..., help="image_dir"),
    out_dir: Path = typer.Argument(..., help="image_dir"),
    nprocs: int = typer.Option(24, help="Nprocs"),
):
    npy_paths = get_filelist_and_cache(npy_dir, "*.npy")
    with Pool(nprocs) as p:
        list(
            tqdm(
                p.imap(
                    partial(
                        atomic_draw_lmk_mediapipe,
                        npy_dir=npy_dir,
                        out_dir=out_dir,
                    ),
                    npy_paths,
                ),
                total=len(npy_paths),
            )
        )


if __name__ == "__main__":
    app()

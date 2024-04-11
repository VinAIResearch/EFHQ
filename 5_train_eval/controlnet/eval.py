import shutil
from pathlib import Path

import cv2
import numpy as np
import typer
from path_utils import get_filelist_and_cache
from tqdm.rich import tqdm


def L2(p1, p2):
    return np.linalg.norm(p1 - p2)


def NME(landmarks_gt, landmarks_pv):
    pts_num = landmarks_gt.shape[0]
    if pts_num == 29:
        left_index = 16
        right_index = 17
    elif pts_num == 68:
        left_index = 36
        right_index = 45
    elif pts_num == 98:
        left_index = 60
        right_index = 72
    elif pts_num == 478:
        left_index = 33
        right_index = 263

    nme = 0
    eye_span = L2(landmarks_gt[left_index], landmarks_gt[right_index])
    for i in range(pts_num):
        error = L2(landmarks_pv[i], landmarks_gt[i])
        nme += error / eye_span
    nme /= pts_num
    return nme


def get_best_lmk(landmarkss):
    biggest = 0
    biggest_idx = 0
    assert len(landmarkss.shape) == 3, landmarkss.shape
    if landmarkss.shape[0] == 1:
        return landmarkss[0]
    for j, landmarks in enumerate(landmarkss):
        face_rect = [
            landmarks[0][0],
            landmarks[0][1],
            landmarks[0][0],
            landmarks[0][1],
        ]  # Left, up, right, down.
        for i in range(len(landmarks)):
            face_rect[0] = min(face_rect[0], landmarks[i][0])
            face_rect[1] = min(face_rect[1], landmarks[i][1])
            face_rect[2] = max(face_rect[2], landmarks[i][0])
            face_rect[3] = max(face_rect[3], landmarks[i][1])
        w = face_rect[2] - face_rect[0]
        h = face_rect[3] - face_rect[1]
        area = w * h
        if area > biggest:
            biggest_idx = j
            biggest = area
    return landmarkss[biggest_idx]


app = typer.Typer()


@app.command()
def nme(
    src_path: Path = typer.Argument(..., help="Src path", dir_okay=True, exists=True),
    tgt_path: Path = typer.Argument(..., help="tgt path", dir_okay=True, exists=True),
    dry: bool = typer.Option(False, help="dry"),
    debug_path: Path = typer.Option(Path("./debugNME"), help="debug for drymode"),
):
    src_npys = get_filelist_and_cache(src_path, "*.npy")
    nme_sum = 0
    counter = 0
    if debug_path.exists():
        shutil.rmtree(debug_path)
    debug_path.mkdir(exist_ok=True, parents=True)
    pbar = tqdm(src_npys)
    for i, src_npy_path in enumerate(pbar):
        if counter > 0:
            pbar.set_description(f"{nme_sum/counter}")
        if dry and counter > 100:
            break
        src_lmk = np.load(src_npy_path).astype(np.int)
        src_lmk = get_best_lmk(src_lmk)

        target_folder = tgt_path / src_npy_path.parent.relative_to(src_path) / src_npy_path.stem
        if not target_folder.exists():
            target_folder = tgt_path / src_npy_path.stem
        if not target_folder.exists():
            continue
        tgt_npys = list(target_folder.glob("*.npy"))
        for j, tgt_npy_path in enumerate(tgt_npys):
            if tgt_npy_path.stem == "condition":
                continue
            tgt_lmk = np.load(tgt_npy_path).reshape(-1, src_lmk.shape[0], 2)
            tgt_lmk = get_best_lmk(tgt_lmk)

            nme = NME(src_lmk, tgt_lmk)

            if dry or nme > 0.5:
                dop = debug_path / src_npy_path.stem / f"{tgt_npy_path.stem}.jpg"
                dop.parent.mkdir(exist_ok=True, parents=True)
                blank = np.zeros((512, 512, 3), dtype=np.uint8)
                for sl, tl in zip(src_lmk, tgt_lmk):
                    cv2.circle(blank, (sl[0], sl[1]), 1, (0, 255, 0), -1)
                    cv2.circle(blank, (tl[0], tl[1]), 1, (0, 0, 255), -1)
                cv2.imwrite(dop.as_posix(), blank)
            nme_sum += nme
            counter += 1
    print(f"NME={nme_sum/counter}")


if __name__ == "__main__":
    app()

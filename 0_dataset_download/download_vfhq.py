import argparse
import glob
import os
from pathlib import Path

import cv2
import youtube_dl
import yt_dlp
from tqdm import tqdm


format_list = ["webm", "mp4", "avi", "flv", "mkv", "mov"]


def extract_clip_cropped_face(clip_meta_path, vid_path, save_vid_root, save_cropped_face_root):
    # read the basic info
    if not os.path.exists(clip_meta_path):
        print("File not found: ", clip_meta_path)
        return
    clip_meta_file = open(clip_meta_path)
    clip_name = os.path.splitext(os.path.basename(clip_meta_path))[0]
    for line in clip_meta_file:
        if line.startswith("H"):
            clip_height = int(line.strip().split(" ")[-1])
        if line.startswith("W"):
            clip_width = int(line.strip().split(" ")[-1])
        if line.startswith("FPS"):
            clip_fps = float(line.strip().split(" ")[-1])
        # get the coordinates of face
        if line.startswith("CROP"):
            clip_crop_bbox = line.strip().split(" ")[-4:]
            x0 = int(clip_crop_bbox[0])
            y0 = int(clip_crop_bbox[1])
            x1 = int(clip_crop_bbox[2])
            y1 = int(clip_crop_bbox[3])

    _, _, pid, clip_idx, frame_rlt = clip_name.split("+")

    save_cropped_face_clip_root = os.path.join(save_cropped_face_root, pid, clip_idx)
    os.makedirs(save_cropped_face_clip_root, exist_ok=True)

    pid = int(pid.split("P")[1])
    clip_idx = int(clip_idx.split("C")[1])
    frame_start, frame_end = frame_rlt.replace("F", "").split("-")
    # NOTE
    frame_start, frame_end = int(frame_start) + 1, int(frame_end) - 1

    start_t = round(frame_start / float(clip_fps), 5)
    end_t = round(frame_end / float(clip_fps), 5)
    end_t - start_t

    save_clip_root = os.path.join(save_vid_root, clip_name)
    os.makedirs(save_clip_root, exist_ok=True)

    try:
        ffmpeg_cmd = (
            rf'ffmpeg -nostdin -loglevel error -i {vid_path} -an -vf "select=between(n\,{frame_start}\,{frame_end}),setpts=PTS-STARTPTS" '  # noqa E501
            f"-qscale:v 1 -qmin 1 -qmax 1 -vsync 0 -start_number {frame_start} "
            f"{save_clip_root}/%08d.png"
        )
        os.system(ffmpeg_cmd)
    except Exception as e:
        print(f"{vid_path}: {e}")

    # crop the HQ frames
    hq_frame_list = sorted(glob.glob(os.path.join(save_clip_root, "*")))
    if len(hq_frame_list) > 0:
        for frame_path in hq_frame_list:
            try:
                basename = os.path.splitext(os.path.basename(frame_path))[0]
                frame = cv2.imread(frame_path)
                frame_height, frame_width, _ = frame.shape
                if frame_height != clip_height or frame_width != clip_width:
                    print("Resizing: ", vid_path)
                    frame = cv2.resize(frame, (clip_width, clip_height))
                cropped_face = frame[y0:y1, x0:x1]
                final_path = os.path.join(save_cropped_face_clip_root, f"{basename}.png")
                cv2.imwrite(final_path, cropped_face)
            except Exception as e:
                print(f"{frame_path}: {e}")
                continue


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="Crawling VFHQ Dataset")
    parser.add_argument("--annotation", dest="annotation", help="Text File of Video ID", default="", type=str)
    parser.add_argument(
        "--annotation_folder",
        dest="annotation_folder",
        help="Annotation folder",
        default="",
        type=str,
    )
    parser.add_argument(
        "--save_folder",
        dest="save_folder",
        help="Save downloaded clips to folder",
        default=False,
        type=str,
    )
    parser.add_argument("--check_point", dest="check_point", help="check_point folder", default=False, type=str)
    parser.add_argument("--type", dest="type", help="Train/Test", default=False, type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    save_folder_path = os.path.join(args.save_folder, "videos", args.type)
    os.makedirs(save_folder_path, exist_ok=True)

    with open(args.annotation) as f:
        annotation_id_list = f.readlines()
        f.close()
    annotation_id_list = [i.strip() for i in annotation_id_list]

    if os.path.exists(args.check_point):
        if os.stat(args.check_point).st_size == 0:
            done_id_list = []
        else:
            check_point_read = open(args.check_point)
            done_id_list = check_point_read.readlines()
            check_point_read.close()
            done_id_list = [i.strip() for i in done_id_list]
        annotation_id_list = [i for i in annotation_id_list if i not in done_id_list]

    pbar = tqdm(annotation_id_list)
    for annotation_id in pbar:
        clip_name = os.path.basename(annotation_id)
        _, video_id, pid, clip_idx, frame_rlt = clip_name.split("+")

        vid_path = None
        for format_type in format_list:
            if os.path.exists(os.path.join(save_folder_path, f"{video_id}.{format_type}")):
                vid_path = os.path.join(save_folder_path, f"{video_id}.{format_type}")
                break
            else:
                continue

        if vid_path is None:
            try_again = False
            ydl_opts = {
                "format": "bestvideo",
                "outtmpl": f"{save_folder_path}/%(id)s.%(ext)s",
                "ignore-errors": True,
                "retries": 10,
                "fragment-retries": 10,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                try:
                    ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
                except Exception as error:
                    print("First try Error: ", error, annotation_id)
                    try_again = True
                if try_again:
                    ydl_opts2 = {
                        "format": "bestvideo/best",
                        "outtmpl": f"{save_folder_path}/%(id)s.%(ext)s",
                        "ignore-errors": True,
                    }
                    with youtube_dl.YoutubeDL(ydl_opts2) as ydl2:
                        try:
                            ydl2.download([f"https://www.youtube.com/watch?v={video_id}"])
                        except Exception as error:
                            print("Second try Error: ", error, annotation_id)
            vid_path = ydl_opts["outtmpl"]
        if vid_path is not None:
            # save extracted hq frames
            save_extracted_hq_results = f"processed/extracted_hq_results/{args.type}/{video_id}"
            # save cropped hq faces
            save_cropped_face_root = f"processed/extracted_cropped_face_results/{args.type}/{video_id}"

            # Create folders
            save_extracted_hq_results_path = os.path.join(args.save_folder, save_extracted_hq_results)
            save_cropped_face_root_path = os.path.join(args.save_folder, save_cropped_face_root)
            os.makedirs(save_extracted_hq_results_path, exist_ok=True)
            os.makedirs(save_cropped_face_root_path, exist_ok=True)

            clip_meta_path = os.path.join(args.annotation_folder, f"{args.type}", f"{annotation_id}.txt")
            print(clip_meta_path)
            extract_clip_cropped_face(
                clip_meta_path,
                vid_path,
                save_extracted_hq_results_path,
                save_cropped_face_root_path,
            )
        if os.path.exists(args.check_point):
            if os.stat(args.check_point).st_size == 0:
                check_point = open(args.check_point, "w")
            else:
                check_point = open(args.check_point, "a")
        else:
            p = Path(args.check_point)
            p.parent.mkdir(exist_ok=True, parents=True)
            check_point = open(args.check_point, "w")
        check_point.write(f"{annotation_id}\n")
        check_point.close()

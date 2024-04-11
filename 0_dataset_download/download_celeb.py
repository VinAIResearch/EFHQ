import argparse
import json
import os

import cv2
from tqdm import tqdm


def download(video_path, ytb_id, proxy=None):
    """
    ytb_id: youtube_id
    save_folder: save video folder
    proxy: proxy url, default None
    """
    if proxy is not None:
        proxy_cmd = f"--proxy {proxy}"
    else:
        proxy_cmd = ""
    if not os.path.exists(video_path):
        down_video = " ".join(
            [
                "yt-dlp",
                proxy_cmd,
                "-f",
                "'bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio'",
                "--skip-unavailable-fragments",
                "--merge-output-format",
                "mp4",
                "https://www.youtube.com/watch?v=" + ytb_id,
                "--output",
                video_path,
                "--external-downloader",
                "aria2c",
                "--external-downloader-args",
                '"-x 16 -k 1M"',
            ]
        )
        print(down_video)
        status = os.system(down_video)
        if status != 0:
            print(f"video not found: {ytb_id}")


def process_ffmpeg(raw_vid_path, save_folder, save_vid_name, bbox, time):
    """
    raw_vid_path:
    save_folder:
    save_vid_name:
    bbox: format: top, bottom, left, right. the values are normalized to 0~1
    time: begin_sec, end_sec
    """

    def secs_to_timestr(secs):
        hrs = secs // (60 * 60)
        min = (secs - hrs * 3600) // 60  # thanks @LeeDongYeun for finding & fixing this bug
        sec = secs % 60
        end = (secs - int(secs)) * 100
        return f"{int(hrs):02d}:{int(min):02d}:{int(sec):02d}.{int(end):02d}"

    def expand(bbox, ratio):
        top, bottom = max(bbox[0] - ratio, 0), min(bbox[1] + ratio, 1)
        left, right = max(bbox[2] - ratio, 0), min(bbox[3] + ratio, 1)

        return top, bottom, left, right

    def to_square(bbox):
        top, bottom, leftx, right = bbox
        h = bottom - top
        w = right - leftx
        c = min(h, w) // 2
        c_h = (top + bottom) / 2
        c_w = (leftx + right) / 2

        top, bottom = c_h - c, c_h + c
        leftx, right = c_w - c, c_w + c
        return top, bottom, leftx, right

    def denorm(bbox, height, width):
        top, bottom, left, right = (
            round(bbox[0] * height),
            round(bbox[1] * height),
            round(bbox[2] * width),
            round(bbox[3] * width),
        )

        return top, bottom, left, right

    out_path = os.path.join(save_folder, save_vid_name.split(".mp4")[0])
    os.makedirs(out_path, exist_ok=True)

    cap = cv2.VideoCapture(raw_vid_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    top, bottom, left, right = to_square(denorm(expand(bbox, 0.02), height, width))
    start_sec, end_sec = time

    cmd = f"ffmpeg -i {raw_vid_path} -vf crop=w={right-left}:h={bottom-top}:x={left}:y={top} -ss {secs_to_timestr(start_sec)} -to {secs_to_timestr(end_sec)} -loglevel error {out_path}/%08d.png"  # noqa
    os.system(cmd)
    return out_path


def load_data(file_path):
    with open(file_path) as f:
        data_dict = json.load(f)

    for key, val in data_dict["clips"].items():
        save_name = key + ".mp4"
        ytb_id = val["ytb_id"]
        time = val["duration"]["start_sec"], val["duration"]["end_sec"]

        bbox = [
            val["bbox"]["top"],
            val["bbox"]["bottom"],
            val["bbox"]["left"],
            val["bbox"]["right"],
        ]
        yield ytb_id, save_name, time, bbox


def load_item(data_dict, key):
    save_name = key + ".mp4"
    val = data_dict["clips"][key]
    ytb_id = val["ytb_id"]
    time = val["duration"]["start_sec"], val["duration"]["end_sec"]

    bbox = [val["bbox"]["top"], val["bbox"]["bottom"], val["bbox"]["left"], val["bbox"]["right"]]
    return ytb_id, save_name, time, bbox


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="Crawling VFHQ Dataset")
    parser.add_argument("--annotation", dest="annotation", help="Text File of Video ID", default="", type=str)
    parser.add_argument("--metadata", dest="metadata", help="metadata path", default="celebvhq_info.json", type=str)
    parser.add_argument(
        "--save_folder",
        dest="save_folder",
        help="Save downloaded clips to folder",
        default=False,
        type=str,
    )
    parser.add_argument("--process", dest="process", required=True, type=int)
    parser.add_argument("--check_point", dest="check_point", help="check_point folder", default=False, type=str)
    parser.add_argument("--type", dest="type", help="Train/Test", default="all", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    raw_vid_root = os.path.join(args.save_folder, "videos", args.type)
    processed_vid_root = os.path.join(args.save_folder, f"processed/extracted_hq_results/{args.type}")
    os.makedirs(raw_vid_root, exist_ok=True)
    os.makedirs(processed_vid_root, exist_ok=True)

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

    proxy = None  # proxy url example, set to None if not use
    with open(args.metadata) as f:
        data_dict = json.load(f)
        f.close()

    pbar = tqdm(annotation_id_list)
    for item in pbar:
        vid_id, save_vid_name, time, bbox = load_item(data_dict, item)
        raw_vid_path = os.path.join(raw_vid_root, vid_id + ".mp4")

        if args.process == 0:
            download(raw_vid_path, vid_id, proxy)
        else:
            if not os.path.exists(os.path.join(processed_vid_root, save_vid_name)):
                process_ffmpeg(raw_vid_path, processed_vid_root, save_vid_name, bbox, time)

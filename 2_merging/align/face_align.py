import cv2
import numpy as np
import PIL.Image
import scipy.ndimage


def adjust_coordinate(coord, new_shape, old_shape):
    ratio_x = float(old_shape[0] / new_shape[0])
    ratio_y = float(old_shape[1] / new_shape[1])

    coord = coord / np.array([ratio_x, ratio_y])
    return coord


def cropByInputLM(img, lms, rescale):
    nLM = lms.shape[0]
    lms_x = [lms[i, 0] for i in range(0, nLM)]
    lms_y = [lms[i, 1] for i in range(0, nLM)]
    return cropImg(img, min(lms_x), min(lms_y), max(lms_x), max(lms_y), rescale)


def cropImg(img, tlx, tly, brx, bry, rescale):
    l = float(tlx)
    t = float(tly)
    ww = float(brx - l)
    hh = float(bry - t)

    # Approximate LM tight BB
    img.shape[0]
    img.shape[1]

    cx = l + ww / 2
    cy = t + hh / 2
    tsize = max(ww, hh) / 2
    l = cx - tsize
    t = cy - tsize

    # Approximate expanded bounding box
    bl = int(round(cx - rescale[0] * tsize))
    bt = int(round(cy - rescale[1] * tsize))
    br = int(round(cx + rescale[2] * tsize))
    bb = int(round(cy + rescale[3] * tsize))
    int(br - bl)
    int(bb - bt)
    # imcrop = np.zeros((nh, nw, 3), dtype="uint8")

    bbox = [bl, bt, br, bb]
    return bbox


def image_align_68(
    image,
    face_landmarks_5pts,
    face_landmarks_68pts,
    output_size=1024,
    transform_size=4096,
    enable_padding=True,
):
    # Align function from FFHQ dataset pre-processing step
    # https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py
    np.random.seed(12345)
    face_landmarks_ref = face_landmarks_68pts
    bbox = cropByInputLM(image, face_landmarks_ref, rescale=[1.4255, 2.0591, 1.6423, 1.3087])
    cv2_shape = image.shape
    if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > cv2_shape[1] or bbox[3] > cv2_shape[0]:
        return None, None, None

    lm = face_landmarks_ref
    lm_5pts = face_landmarks_5pts
    lm_68pts = face_landmarks_68pts
    lm[0:17]  # left-right
    lm[17:22]  # left-right
    lm[22:27]  # left-right
    lm[27:31]  # top-down
    lm[31:36]  # top-down
    lm_eye_left = lm[36:42]  # left-clockwise
    lm_eye_right = lm[42:48]  # left-clockwise
    lm_mouth_outer = lm[48:60]  # left-clockwise
    lm[60:68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    img = PIL.Image.fromarray(image)
    original_size = img.size

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (
            int(np.rint(float(img.size[0]) / shrink)),
            int(np.rint(float(img.size[1]) / shrink)),
        )
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        lm = adjust_coordinate(lm, img.size, original_size)
        lm_5pts = adjust_coordinate(lm_5pts, img.size, original_size)
        lm_68pts = adjust_coordinate(lm_68pts, img.size, original_size)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (
        int(np.floor(min(quad[:, 0]))),
        int(np.floor(min(quad[:, 1]))),
        int(np.ceil(max(quad[:, 0]))),
        int(np.ceil(max(quad[:, 1]))),
    )
    crop = (
        max(crop[0] - border, 0),
        max(crop[1] - border, 0),
        min(crop[2] + border, img.size[0]),
        min(crop[3] + border, img.size[1]),
    )
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        lm = lm - np.array([crop[0], crop[1]])
        lm_5pts = lm_5pts - np.array([crop[0], crop[1]])
        lm_68pts = lm_68pts - np.array([crop[0], crop[1]])
        quad -= crop[0:2]

    # Pad.
    pad = (
        int(np.floor(min(quad[:, 0]))),
        int(np.floor(min(quad[:, 1]))),
        int(np.ceil(max(quad[:, 0]))),
        int(np.ceil(max(quad[:, 1]))),
    )
    pad = (
        max(-pad[0] + border, 0),
        max(-pad[1] + border, 0),
        max(pad[2] - img.size[0] + border, 0),
        max(pad[3] - img.size[1] + border, 0),
    )
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), "reflect")
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(
            1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
            1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]),
        )
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = np.uint8(np.clip(np.rint(img), 0, 255))
        lm += np.array([pad[0], pad[1]])
        lm_5pts += np.array([pad[0], pad[1]])
        lm_68pts += np.array([pad[0], pad[1]])
        img = PIL.Image.fromarray(img, "RGB")
        quad += pad[:2]

    # Transform.
    cv2_image = np.array(img).copy()
    cv2_image = cv2_image[:, :, ::-1].copy()

    target = np.array(
        [(0, 0), (0, transform_size), (transform_size, transform_size), (transform_size, 0)],
        np.float32,
    )
    M = cv2.getPerspectiveTransform(np.float32(quad + 0.5), target)
    transformed_image = cv2.warpPerspective(cv2_image, M, (transform_size, transform_size), cv2.INTER_LINEAR)

    lm = cv2.perspectiveTransform(np.expand_dims(lm, axis=1), M)  # Adjust landmarks
    lm = np.squeeze(lm, 1)

    lm_5pts = cv2.perspectiveTransform(np.expand_dims(lm_5pts, axis=1), M)  # Adjust landmarks
    lm_5pts = np.squeeze(lm_5pts, 1)

    lm_68pts = cv2.perspectiveTransform(np.expand_dims(lm_68pts, axis=1), M)  # Adjust landmarks
    lm_68pts = np.squeeze(lm_68pts, 1)

    img = img.transform(
        (transform_size, transform_size),
        PIL.Image.QUAD,
        (quad + 0.5).flatten(),
        PIL.Image.BILINEAR,
    )
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

        original_size = transformed_image.shape
        transformed_image = cv2.resize(transformed_image, (output_size, output_size))
        lm = adjust_coordinate(lm, transformed_image.shape, original_size)
        lm_5pts = adjust_coordinate(lm_5pts, transformed_image.shape, original_size)
        lm_68pts = adjust_coordinate(lm_68pts, transformed_image.shape, original_size)

    img_np = np.array(img)
    return img_np, lm_5pts, lm_68pts


def image_align_5(
    image,
    face_landmarks_5pts,
    face_landmarks_68pts,
    output_size=1024,
    transform_size=4096,
    enable_padding=True,
):
    # Align function from FFHQ dataset pre-processing step
    # https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py
    np.random.seed(12345)
    face_landmarks_ref = face_landmarks_5pts
    bbox = cropByInputLM(image, face_landmarks_ref, rescale=[1.4255, 2.0591, 1.6423, 1.3087])
    cv2_shape = image.shape
    if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > cv2_shape[1] or bbox[3] > cv2_shape[0]:
        return None, None, None

    lm = face_landmarks_ref
    lm_5pts = face_landmarks_5pts
    lm_68pts = face_landmarks_68pts
    lm_eye_left = lm[0]  # left-clockwise
    lm_eye_right = lm[1]  # left-clockwise
    # lm_nose = lm[2]
    lm_mouth_left = lm[3]  # left-clockwise
    lm_mouth_right = lm[4]  # left-clockwise

    eye_avg = (lm_eye_left + lm_eye_right) * 0.5
    eye_to_eye = lm_eye_right - lm_eye_left
    mouth_avg = (lm_mouth_left + lm_mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    img = PIL.Image.fromarray(image)
    original_size = img.size

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (
            int(np.rint(float(img.size[0]) / shrink)),
            int(np.rint(float(img.size[1]) / shrink)),
        )
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        lm = adjust_coordinate(lm, img.size, original_size)
        lm_5pts = adjust_coordinate(lm_5pts, img.size, original_size)
        lm_68pts = adjust_coordinate(lm_68pts, img.size, original_size)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (
        int(np.floor(min(quad[:, 0]))),
        int(np.floor(min(quad[:, 1]))),
        int(np.ceil(max(quad[:, 0]))),
        int(np.ceil(max(quad[:, 1]))),
    )
    crop = (
        max(crop[0] - border, 0),
        max(crop[1] - border, 0),
        min(crop[2] + border, img.size[0]),
        min(crop[3] + border, img.size[1]),
    )
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        lm = lm - np.array([crop[0], crop[1]])
        lm_5pts = lm_5pts - np.array([crop[0], crop[1]])
        lm_68pts = lm_68pts - np.array([crop[0], crop[1]])
        quad -= crop[0:2]

    # Pad.
    pad = (
        int(np.floor(min(quad[:, 0]))),
        int(np.floor(min(quad[:, 1]))),
        int(np.ceil(max(quad[:, 0]))),
        int(np.ceil(max(quad[:, 1]))),
    )
    pad = (
        max(-pad[0] + border, 0),
        max(-pad[1] + border, 0),
        max(pad[2] - img.size[0] + border, 0),
        max(pad[3] - img.size[1] + border, 0),
    )
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), "reflect")
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(
            1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
            1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]),
        )
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = np.uint8(np.clip(np.rint(img), 0, 255))
        lm += np.array([pad[0], pad[1]])
        lm_5pts += np.array([pad[0], pad[1]])
        lm_68pts += np.array([pad[0], pad[1]])
        img = PIL.Image.fromarray(img, "RGB")
        quad += pad[:2]

    # Transform.
    cv2_image = np.array(img).copy()
    cv2_image = cv2_image[:, :, ::-1].copy()

    target = np.array(
        [(0, 0), (0, transform_size), (transform_size, transform_size), (transform_size, 0)],
        np.float32,
    )
    M = cv2.getPerspectiveTransform(np.float32(quad + 0.5), target)
    transformed_image = cv2.warpPerspective(cv2_image, M, (transform_size, transform_size), cv2.INTER_LINEAR)

    lm = cv2.perspectiveTransform(np.expand_dims(lm, axis=1), M)  # Adjust landmarks
    lm = np.squeeze(lm, 1)

    lm_5pts = cv2.perspectiveTransform(np.expand_dims(lm_5pts, axis=1), M)  # Adjust landmarks
    lm_5pts = np.squeeze(lm_5pts, 1)

    lm_68pts = cv2.perspectiveTransform(np.expand_dims(lm_68pts, axis=1), M)  # Adjust landmarks
    lm_68pts = np.squeeze(lm_68pts, 1)

    img = img.transform(
        (transform_size, transform_size),
        PIL.Image.QUAD,
        (quad + 0.5).flatten(),
        PIL.Image.BILINEAR,
    )
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

        original_size = transformed_image.shape
        transformed_image = cv2.resize(transformed_image, (output_size, output_size))
        lm = adjust_coordinate(lm, transformed_image.shape, original_size)
        lm_5pts = adjust_coordinate(lm_5pts, transformed_image.shape, original_size)
        lm_68pts = adjust_coordinate(lm_68pts, transformed_image.shape, original_size)

    img_np = np.array(img)
    return img_np, lm_5pts, lm_68pts

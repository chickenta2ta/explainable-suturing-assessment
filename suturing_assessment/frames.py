import cv2

# Segmentation runs at 10 fps on frame_index % 3 == 0
FRAME_STEP = 3

# 320x240 videos are upscaled to the size of the others so that all coordinates share one scale
FRAME_SIZE = (640, 480)


def video_path(video_dir, video):
    """Path of the capture1 video"""
    return video_dir / f"{video}_capture1.avi"


def read_frames(path, frame_indices):
    """Return {frame_index: image resized to FRAME_SIZE}, decoding from the start to keep frame_index exact"""
    wanted = set(frame_indices)
    frames = {}
    capture = cv2.VideoCapture(str(path))
    index = 0
    while wanted - frames.keys():
        ok, image = capture.read()
        if not ok:
            break
        if index in wanted:
            if image.shape[1::-1] != FRAME_SIZE:
                image = cv2.resize(image, FRAME_SIZE, interpolation=cv2.INTER_CUBIC)
            frames[index] = image
        index += 1
    return frames


def original_size(path):
    """(width, height) of the video"""
    capture = cv2.VideoCapture(str(path))
    return (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )


def count_frames(path):
    """Number of decodable frames (CAP_PROP_FRAME_COUNT can be larger)"""
    capture = cv2.VideoCapture(str(path))
    count = 0
    while capture.grab():
        count += 1
    return count

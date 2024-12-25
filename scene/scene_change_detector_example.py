import argparse
import cv2
from scene_change_detector import SceneChangeDetector


def get_args():
    parser = argparse.ArgumentParser("Example of scene change detector usage")
    parser.add_argument(
        "-v",
        "--video",
        type=str,
        help="Path to the video file.",
        required=True
    )

    args = parser.parse_args()
    return args


def main(opt):
    FLOW_ACTIVATION_THRESHOLD = 0.1
    FLOW_RESIZE = (480, 270)
    FRAMES_DIFF = 10

    video_path = opt.video

    scene_detector = SceneChangeDetector(FRAMES_DIFF, FLOW_ACTIVATION_THRESHOLD, FLOW_RESIZE)

    cap = cv2.VideoCapture(video_path)
    counter = 0

    while True:
        ret, frame = cap.read()

        if frame is None:
            break

        activation = scene_detector.process(frame)

        if activation:
            print('activation', counter)

        cv2.imshow('frame', frame)
        c = cv2.waitKey(1)

        if c == 27:
            break
        elif c == 32:
            while True:
                if cv2.waitKey(1) == 32:
                    break

        counter += 1


if __name__ == "__main__":
    options = get_args()
    main(options)

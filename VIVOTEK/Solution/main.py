import argparse
import cv2
import os
import json
from transformers import Swinv2ForImageClassification, Swinv2Config
from transformers import AutoImageProcessor, SwinForImageClassification
from PIL import Image
import torch


def parse_option() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        '2024 spring Deep learning for medical imaging final project',
        add_help=False)
    parser.add_argument('--output_dir',
                        type=str,
                        default='./results',
                        help='output directory')
    parser.add_argument('--seed', type=int, default=1216, help='random seed')

    args, unparsed = parser.parse_known_args()
    return args


def find_continuous_opens(data):
    open_ranges = []
    start = None
    count_closed = 0

    for i, status in enumerate(data):
        if status == 'Open':
            if start is None:
                start = i
            count_closed = 0
        else:
            count_closed += 1
            if count_closed > 2:  # if more than 2 consecutive closed
                if start is not None:
                    open_ranges.append((start, i - count_closed))
                    start = None
                count_closed = 0

    if start is not None:
        open_ranges.append([start, len(data) - count_closed - 1])

    return open_ranges


def guess(video_filename, return_type):
    args = parse_option()
    model = Swinv2ForImageClassification.from_pretrained(
        os.path.join(args.output_dir, "image"))
    image_processor = AutoImageProcessor.from_pretrained(
        "microsoft/swin-tiny-patch4-window7-224")

    result = list()
    cap = cv2.VideoCapture(os.path.join('../Tests', video_filename))

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        inputs = image_processor(pil_image, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_label = logits.argmax(-1).item()
        result.append(model.config.id2label[predicted_label])

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    result = find_continuous_opens(result)

    if not result:
        return 0
    elif return_type == 'opening':
        return result[0][0]
    elif return_type == 'closing':
        return result[-1][1]
    else:
        return None


def guess_door_opening(video_filename):
    """ Simulate guessing the frame for door opening. """
    # Hypothetical function: replace with actual logic.
    # return 100  # Placeholder value, should be replaced with real computation based on video analysis.
    return guess(video_filename, 'opening')


def guess_door_closing(video_filename):
    """ Simulate guessing the frame for door closing. """
    # Hypothetical function: replace with actual logic.
    # return 200  # Placeholder value, should be replaced with real computation based on video analysis.
    return guess(video_filename, 'closing')


def scan_videos(directory):
    """Scan the specified directory for MP4 files and generate JSON annotations."""
    video_files = [f for f in os.listdir(directory) if f.endswith('.mp4')]
    videos_info = []

    for video_file in video_files:
        videos_info.append({
            "video_filename":
            video_file,
            "annotations": [{
                "object":
                "Door",
                "states": [
                    {
                        "state_id": 1,
                        "description": "Opening",
                        "guessed_frame": guess_door_opening(
                            video_file)  # Guessing frame using function.
                    },
                    {
                        "state_id": 2,
                        "description": "Closing",
                        "guessed_frame": guess_door_closing(
                            video_file)  # Guessing frame using function.
                    }
                ]
            }]
        })

    return videos_info


def generate_json(output_filename, videos_info):
    """Generate a JSON file with the provided video information."""
    with open(output_filename, 'w') as file:
        json.dump({"videos": videos_info}, file, indent=4)


def main():
    directory = "./Tests"
    output_filename = "output.json"
    videos_info = scan_videos(directory)
    generate_json(output_filename, videos_info)
    print(f"Generated JSON file '{output_filename}' with video annotations.")


if __name__ == "__main__":
    main()

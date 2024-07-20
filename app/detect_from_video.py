import os
import argparse
from os.path import join
import cv2
import dlib
import torch
import torch.nn as nn
from PIL import Image as pil_image
from tqdm import tqdm
from network.models import model_selection
from dataset.transform import xception_default_data_transforms


def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (x1 + x2) // 2
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)
    return x1, y1, size_bb


def preprocess_image(image, cuda=True):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    preprocess = xception_default_data_transforms['test']
    preprocessed_image = preprocess(pil_image.fromarray(image))
    preprocessed_image = preprocessed_image.unsqueeze(0)
    if cuda and torch.cuda.is_available():
        preprocessed_image = preprocessed_image.cuda()
    return preprocessed_image


def predict_with_model(image, model, post_function=nn.Softmax(dim=1), cuda=True):
    preprocessed_image = preprocess_image(image, cuda)
    output = model(preprocessed_image)
    output = post_function(output)
    _, prediction = torch.max(output, 1)
    prediction = float(prediction.cpu().numpy())
    return int(prediction), output


def test_full_image_network(video_path, model_path, output_path, start_frame=0, end_frame=None, cuda=True):
    print('Starting: {}'.format(video_path))
    reader = cv2.VideoCapture(video_path)
    video_fn = video_path.split('/')[-1].split('.')[0] + '_output.avi'
    output_filepath = join(output_path, video_fn)
    os.makedirs(output_path, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = reader.get(cv2.CAP_PROP_FPS)
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = None
    face_detector = dlib.get_frontal_face_detector()

    model, *_ = model_selection(modelname='xception', num_out_classes=2)
    if model_path and os.path.exists(model_path):
        try:
            model = torch.load(model_path)
            print('Model loaded from {}'.format(model_path))
        except Exception as e:
            print(f"Failed to load model from {model_path}: {e}")
            print("Using pre-trained weights instead.")
    else:
        print('No valid model file provided or found, using pre-trained weights instead.')
    if cuda and torch.cuda.is_available():
        model = model.cuda()
    else:
        print("CUDA not available, running on CPU")

    font_face = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    font_scale = 1
    frame_num = 0
    assert start_frame < num_frames - 1
    end_frame = end_frame if end_frame else num_frames
    pbar = tqdm(total=end_frame - start_frame)
    while reader.isOpened():
        ret, image = reader.read()
        if not ret:
            break
        frame_num += 1
        if frame_num < start_frame:
            continue
        pbar.update(1)
        height, width = image.shape[:2]
        if writer is None:
            writer = cv2.VideoWriter(output_filepath, fourcc, fps, (width, height))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 1)
        if len(faces):
            face = faces[0]
            x, y, size = get_boundingbox(face, width, height)
            cropped_face = image[y:y + size, x:x + size]
            prediction, output = predict_with_model(cropped_face, model, cuda=cuda)
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y
            label = 'fake' if prediction == 1 else 'real'
            color = (0, 255, 0) if prediction == 0 else (0, 0, 255)
            output_list = ['{0:.2f}'.format(float(x)) for x in output.detach().cpu().numpy()[0]]
            cv2.putText(image, str(output_list) + '=>' + label, (x, y + h + 30), font_face, font_scale, color, thickness, 2)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        if frame_num >= end_frame:
            break
        writer.write(image)  # Write the frame to the output video
    pbar.close()
    if writer is not None:
        writer.release()
        print('Finished! Output saved at {}'.format(output_filepath))
    else:
        print('Input video file was empty')


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--video_path', '-i', type=str)
    p.add_argument('--model_path', '-m', type=str, default=None)
    p.add_argument('--output_path', '-o', type=str, default='.')
    p.add_argument('--start_frame', type=int, default=0)
    p.add_argument('--end_frame', type=int, default=None)
    p.add_argument('--cuda', action='store_true')
    args = p.parse_args()

    video_path = args.video_path
    if video_path.endswith('.mp4') or video_path.endswith('.avi'):
        test_full_image_network(**vars(args))
    else:
        videos = os.listdir(video_path)
        for video in videos:
            args.video_path = join(video_path, video)
            test_full_image_network(**vars(args))


if __name__ == '__main__':
    main()

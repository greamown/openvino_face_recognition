import json, os, shutil, sys, cv2
import argparse
from pathlib import Path
from time import perf_counter
from common import read_json
from common import config_logger 
from model_api import Detection, draw_box

# def draw_box(info, palette):
#     class_id = 0
#     frame = info["frame"]
#     output_transform = info['output_transform']
#     if info["detections"] is not None and info["detections"] != []:
#         if output_transform != None:
#             frame = output_transform.resize(frame)
#         boxes = info["detections"][0]
#         for box in boxes:
#             xmin, ymin, xmax, ymax = output_transform.scale([box[0], box[1], box[2], box[3]])
#             cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), palette[class_id], 2)
#     return frame

def main(args):
    # Setting config
    config = read_json(args.config)
    # Source
    from common.images_capture import open_images_capture
    cap = open_images_capture(config['source'], config['loop'])
    # Loading model
    det = Detection(config)
    palette = det.load_model() 

    while True:
        frame = cap.read()
        info = det.inference(frame)
        # ---------------------------Drawing detecter to information-----------------------------------------------------------------------
        if info is not None:
            frame = draw_box(info, draw_key=True, palette=palette)
        # ---------------------------Show--------------------------------------------------------------------------------------------------              
        cv2.imshow('Detection Results', frame)
        key = cv2.waitKey(1)
        ESC_KEY = 27
        # Quit.
        if key in {ord('q'), ord('Q'), ESC_KEY}:
            break

if __name__ == '__main__':
    config_logger('./openvino_face_detection.log', 'w', "info")
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help = "The path of application config")
    args = parser.parse_args()
    sys.exit(main(args) or 0)
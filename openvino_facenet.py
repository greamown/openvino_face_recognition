import json, os, shutil, sys, cv2
import argparse
from pathlib import Path
from time import perf_counter
from common import read_json, write_json
from common import config_logger 
from model_api import Recognition

def main(args):
    # Setting config
    config = read_json(args.config)
    # Source
    from common import open_images_capture
    cap = open_images_capture(config['source'], config['loop'])
    # Loading model
    det = Recognition(config)
    det.load_model()

    while True:
        frame = cap.read()
        info = det.inference(frame)
        # ---------------------------Drawing detecter to information-----------------------------------------------------------------------
        if info is not None:
            path_list = os.path.split(config["source"])
            main_path = path_list[0]
            img_name = path_list[-1]
            json_path = os.path.join(main_path, "feature.json")
            if os.path.exists(json_path):
                content = read_json(json_path)
                key = os.path.splitext(img_name)[0]
                char_str = '' .join((z for z in key if not z.isdigit()))
                if char_str in list(content.keys()):
                    content[char_str].append(info["detections"][0][0].tolist())
                    # write_json(json_path, content)
                break
            else:
                print("[{}] has not exist.".format(json_path))

        # ---------------------------Show--------------------------------------------------------------------------------------------------              
    sys.exit(0)
if __name__ == '__main__':
    config_logger('./openvino_facenet.log', 'w', "info")
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help = "The path of application config")
    args = parser.parse_args()
    sys.exit(main(args) or 0)
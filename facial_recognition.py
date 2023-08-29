import os, sys, cv2, glob, time, logging
from argparse import ArgumentParser, SUPPRESS
from common import read_json, ALLOWED_EXTENSIONS
from common import config_logger
from model_api import Recognition, Detection, draw_box
from database import init_db, insert_table_cmd, execute_db

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument('-c', '--config', required=True, help = "The path of application config")
    return parser

def main(args):
    # Setting config
    config = read_json(args.config)
    # Loading model
    det = Detection(config)
    det.load_model() 
    rec = Recognition(config)
    rec.load_model()
    # Reading db

    # Source
    from common.images_capture import open_images_capture
    cap = open_images_capture(config['source'], config['loop'])
    while True:
        frame = cap.read()
        # Detect box
        info = det.inference(frame)
        # Catch face box  and get feature
        if info is not None:
            dets = draw_box(info)
            # Get feature
            info = rec.inference(dets[0])
            if info is not None:
                # SVM/KNN with all db data
                pass
        # ---------------------------Show--------------------------------------------------------------------------------------------------              
        cv2.imshow('Detection Results', frame)
        key = cv2.waitKey(1)
        ESC_KEY = 27
        # Quit.
        if key in {ord('q'), ord('Q'), ESC_KEY}:
            break

if __name__ == '__main__':
    config_logger('./facial_recognition.log', 'w', "info")
    args = build_argparser().parse_args()
    sys.exit(main(args) or 0)
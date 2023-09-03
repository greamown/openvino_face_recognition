import  logging, cv2, sys
import numpy as np
from argparse import ArgumentParser, SUPPRESS
from common import read_json, config_logger
from model_api import Recognition, Detection, draw_box, calcu_distance
from database import read_feature_db

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
    palette = det.load_model() 
    rec = Recognition(config, initial=True)
    rec.load_model()
    # Reading db
    err_db = read_feature_db()
    if err_db:
        logging.error(str(err_db[-1]))
        return 
    
    # Source
    from common.images_capture import open_images_capture
    cap = open_images_capture(config['source'], config['loop'])
    while True:
        frame = cap.read()
        # Detect box
        det_info = det.inference(frame)
        # Catch face box  and get feature
        if det_info is not None:
            dets = draw_box(det_info)
            # Get feature
            names = []
            for de in dets:
                rec_info = rec.inference(de)
                if rec_info is not None:
                    # SVM/KNN with all db data
                    feature = np.array(rec_info["detections"][0][0].tolist())
                    names.append(calcu_distance(feature))
            frame = draw_box(det_info, draw_key=True, palette=palette, names=names)
        # ---------------------------Show--------------------------------------------------------------------------------------------------              
        windowname = 'Detection Results'
        cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)
        cv2.imshow(windowname, frame)
        key = cv2.waitKey(1)
        ESC_KEY = 27
        # Quit.
        if key in {ord('q'), ord('Q'), ESC_KEY}:
            break

    # cv2.destroyAllWindows()
    # sys.exit(0)
    
if __name__ == '__main__':
    config_logger('./facial_recognition.log', 'w', "info")
    args = build_argparser().parse_args()
    sys.exit(main(args) or 0)
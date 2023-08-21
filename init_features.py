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
    # Initial db
    init_db()
    # Catch all images name
    image_list = []
    for name in ALLOWED_EXTENSIONS["image"]:
        img_path = glob.glob(os.path.join(config['init_folder_path'],"*.{}".format(name)))
        image_list += img_path
    # Catch feature and save
    for path in image_list:
        path_list = os.path.split(path)
        main_path = path_list[0]
        img_name = path_list[-1]
        # Read image
        frame = cv2.imread(path)
        # Detect box
        info = det.inference(frame)
        # Catch face box  and get feature
        if info is not None:
            dets = draw_box(info)
            # Get feature
            info = rec.inference(dets[0])
            if info is not None:
                filename = os.path.splitext(img_name)[0]
                feature = str(info["detections"][0][0].tolist())
                create_time = str(time.time())
                # Check data isnot at db
                exist_str = """ SELECT EXISTS( SELECT * FROM feature WHERE name=\'{}\') """.format(filename)
                exist_result = execute_db(exist_str, update=False)
                if not exist_result[0][0]:
                    # Save feature at db
                    error_info = insert_table_cmd("feature", "name, features, create_time", "\'{}\', \'{}\', \'{}\'".format(filename, feature, create_time))
                    if error_info:
                        logging.error(error_info[1])
                else:
                    logging.warning("The data is exist:[{}]".format(filename))

        if image_list.index(path) == len(image_list)-1:
            logging.info("Finished to collect features.")
            os._exit(0)

if __name__ == '__main__':
    config_logger('./init_features.log', 'w', "info")
    args = build_argparser().parse_args()
    sys.exit(main(args) or 0)

import cv2, logging
from openvino.inference_engine import IECore
from common.model import Model
from time import perf_counter
from common.pipelines import get_user_config, AsyncPipeline, Normal
from common.utils import load_labels, resize_image, OutputTransform
import common.box_utils_numpy as box_utils
import numpy as np
import random
import colorsys

class Detection():
    def __init__(self, param, initial=False):
        self.next_frame_id = 0
        self.next_frame_id_to_show = 0
        self.param = param
        self.initial = initial

    def load_model(self):
        # ---------------------------Step 1. Initialize inference engine core--------------------------------------------------
        logging.info('Initializing Inference Engine...')
        ie = IECore()
        # ---------------------------Step 2. Read a model in OpenVINO Intermediate Representation or ONNX format---------------
        self.model = FaceDetection(ie, self.param)
        logging.info('Reading the network: {}'.format(self.param['det_model']))
        logging.info('Loading network...')
        # Get device relative info for inference 
        plugin_config = get_user_config( self.param['device'], flags_nstreams="", flags_nthreads=None)
        # Initialize Pipeline(for inference)
        if self.initial:
            self.detector_pipeline = Normal(ie, self.model, plugin_config,
                                                device=self.param['device'])
        else:
            self.detector_pipeline = AsyncPipeline(ie, self.model, plugin_config,
                                                        device=self.param['device'], max_num_requests=0)
        # ---------------------------Step 3. Create detection words of color---------------
        palette = ColorPalette(len(self.model.labels) if self.model.labels else 100)
        return palette

    def inference(self, frame):
        if self.initial:
            self.submit_action(frame)
        else:
            # Check pipeline is ready & setting output_shape & inference
            if self.detector_pipeline.is_ready():
                self.submit_action(frame)
            else:
                # Wait for empty request
                self.detector_pipeline.await_any()

            if self.detector_pipeline.callback_exceptions:
                raise self.detector_pipeline.callback_exceptions[0]
        # Process all completed requests
        orignal_results = self.detector_pipeline.get_result(self.next_frame_id_to_show)
        
        if orignal_results:
            results = orignal_results[-1]
            results.update({"output_transform": self.output_transform})
            results.update({"detections":orignal_results[0][0]})
            self.next_frame_id_to_show += 1
            return results
        
        return orignal_results
    
    def submit_action(self, frame):
        start_time = perf_counter()
        if frame is None:
            if self.next_frame_id == 0:
                raise ValueError("Can't read an image from the input")
            raise ValueError("Can't read an image")
        if self.next_frame_id == 0:
            # Compute rate from setting output shape and input images shape 
            self.output_transform = OutputTransform(frame.shape[:2], None)  
        # Submit for inference
        self.detector_pipeline.submit_data(frame, self.next_frame_id, {'frame': frame, 'start_time': start_time})
        self.next_frame_id += 1

class FaceDetection(Model):
    def __init__(self, ie, param, labels=None):
        self.model_path = param['det_model']
        self.threshold = param['threshold']

        super().__init__(ie, self.model_path)
        if isinstance(labels, (list, tuple)):
            self.labels = labels
        else:
            self.labels = load_labels(labels) if labels else None

        self.resize_image = resize_image
        
        assert len(self.net.input_info) == 1, "Expected 1 input blob"
        self.image_blob_name = next(iter(self.net.input_info))
        if self.net.input_info[self.image_blob_name].input_data.shape[1] == 3:
            self.n, self.c, self.h, self.w = self.net.input_info[self.image_blob_name].input_data.shape
            self.nchw_shape = True
        else:
            self.n, self.h, self.w, self.c = self.net.input_info[self.image_blob_name].input_data.shape
            self.nchw_shape = False

    def preprocess(self, inputs):
        image = cv2.cvtColor(
            src=inputs,
            code=cv2.COLOR_BGR2RGB,
        )
        # logging.warning('Image is resized from {} to {}'.format(image.shape[:-1],(self.h,self.w)))
        resized_image = self.resize_image(image, (self.w, self.h))
        image_mean = np.array([127, 127, 127])
        resized_image = (resized_image - image_mean) / 128

        meta = {'original_shape': image.shape,
                'resized_shape': resized_image.shape}
        if self.nchw_shape:
            resized_image = resized_image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            resized_image = resized_image.reshape((self.n, self.c, self.h, self.w))

        else:
            resized_image = resized_image.reshape((self.n, self.h, self.w, self.c))

        dict_inputs = {self.image_blob_name: resized_image}

        return dict_inputs, meta

    def postprocess(self, outputs, meta):
        detections = []
        boxes, labels, probs = self.parse_region(meta["original_shape"][1], meta["original_shape"][0], 
                                                    outputs['scores'], outputs['boxes'], self.threshold)
        detections.append([boxes, labels, probs])
        return detections

    def parse_region(self, width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
        boxes = boxes[0]
        confidences = confidences[0]
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, confidences.shape[1]):
            probs = confidences[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.shape[0] == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
            box_probs = box_utils.hard_nms(box_probs,
                                        iou_threshold=iou_threshold,
                                        top_k=top_k,
                                        )
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.shape[0])
        if not picked_box_probs:
            return np.array([]), np.array([]), np.array([])
        picked_box_probs = np.concatenate(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

class ColorPalette:
    def __init__(self, n, rng=None):
        assert n > 0

        if rng is None:
            rng = random.Random(0xACE)

        candidates_num = 100
        hsv_colors = [(1.0, 1.0, 1.0)]
        for _ in range(1, n):
            colors_candidates = [(rng.random(), rng.uniform(0.8, 1.0), rng.uniform(0.5, 1.0))
                                 for _ in range(candidates_num)]
            min_distances = [self.min_distance(hsv_colors, c) for c in colors_candidates]
            arg_max = np.argmax(min_distances)
            hsv_colors.append(colors_candidates[arg_max])

        self.palette = [self.hsv2rgb(*hsv) for hsv in hsv_colors]

    @staticmethod
    def dist(c1, c2):
        dh = min(abs(c1[0] - c2[0]), 1 - abs(c1[0] - c2[0])) * 2
        ds = abs(c1[1] - c2[1])
        dv = abs(c1[2] - c2[2])
        return dh * dh + ds * ds + dv * dv

    @classmethod
    def min_distance(cls, colors_set, color_candidate):
        distances = [cls.dist(o, color_candidate) for o in colors_set]
        return np.min(distances)

    @staticmethod
    def hsv2rgb(h, s, v):
        return tuple(round(c * 255) for c in colorsys.hsv_to_rgb(h, s, v))

    def __getitem__(self, n):
        return self.palette[n % len(self.palette)]

    def __len__(self):
        return len(self.palette)
    
def draw_box(info, draw_key=False, palette=None, names=[]):
    class_id = 0
    frame = info["frame"]
    output_transform = info['output_transform']
    pieces = []
    if info["detections"] is not None and info["detections"] != []:
        if output_transform != None:
            frame = output_transform.resize(frame)
        boxes = info["detections"][0]
        for ind, box in enumerate(boxes):
            axis = output_transform.scale([box[0], box[1], box[2], box[3]])
            xmin, ymin, xmax, ymax = edge_process(axis, frame.shape[:-1])
            if draw_key:
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), palette[class_id], 5)
                if len(names) > 0:
                    cv2.putText(frame, '{}'.format(names[ind]),
                                        (xmin, ymin - 7), cv2.FONT_HERSHEY_COMPLEX, 0.8, palette[-1], 2)
            else:
                pieces.append(frame[ymin:ymax,xmin:xmax])

    if draw_key:
        return frame
    else:
        return pieces
    
def edge_process(axis, shape):
    if axis[0] < 0 :
        axis[0] = 0
    if axis[1] < 0:
        axis[1] = 0
    if axis[2] > shape[1]:
        axis[2] = shape[1]
    if axis[3] > shape[0]:
        axis[3] = shape[0]
    
    return axis

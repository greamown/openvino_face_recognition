from common.model import Model
import cv2, logging
import numpy as np
import random
from time import perf_counter
from common.pipelines import get_user_config, AsyncPipeline
from common.utils import load_labels, resize_image, OutputTransform
from openvino.inference_engine import IECore

class Recognition():
    def __init__(self, param):
        self.next_frame_id = 0
        self.next_frame_id_to_show = 0
        self.param = param

    def load_model(self):
        # ---------------------------Step 1. Initialize inference engine core--------------------------------------------------
        logging.info('Initializing Inference Engine...')
        ie = IECore()
        # ---------------------------Step 2. Read a model in OpenVINO Intermediate Representation or ONNX format---------------
        self.model = Facenet(ie, self.param)
        logging.info('Reading the network: {}'.format(self.param['det_model']))
        logging.info('Loading network...')
        # Get device relative info for inference 
        plugin_config = get_user_config( self.param['device'], flags_nstreams="", flags_nthreads=None)
        # Initialize Pipeline(for inference)
        self.detector_pipeline = AsyncPipeline(ie, self.model, plugin_config,
                                                    device=self.param['device'], max_num_requests=0)

    def inference(self, frame):
        # Check pipeline is ready & setting output_shape & inference
        if self.detector_pipeline.is_ready():
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
        else:
            # Wait for empty request
            self.detector_pipeline.await_any()

        if self.detector_pipeline.callback_exceptions:
            raise self.detector_pipeline.callback_exceptions[0]
        # Process all completed requests
        orignal_results = self.detector_pipeline.get_result(self.next_frame_id_to_show)
        if orignal_results:
            results = {}
            results["detections"] = orignal_results
            results["output_transform"] = self.output_transform
            self.next_frame_id_to_show += 1
            return results
        return orignal_results
    
class Facenet(Model):
    def __init__(self, ie, param):
        self.model_path = param['landmark_model']
        super().__init__(ie, self.model_path)

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
        for output_layer_name in outputs.keys():
            detections.append(outputs[output_layer_name][0])
        return detections
import cv2
import uuid
import os
import time
import wget
import tensorflow as tf
import tkinter
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from google.protobuf import text_format
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('TkAgg')


class TFPP:
    def __init__(self, labels, number_imgs, IMAGES_PATH):
        self.labels = labels
        self.number_imgs = number_imgs
        self.IMAGES_PATH = IMAGES_PATH
        self.LABELIMG_PATH = os.path.join('Tensorflow', 'labelimg')

    def setup_folders(self):
        if not os.path.exists(self.IMAGES_PATH):
            os.mkdir("./" + self.IMAGES_PATH)

        for label in self.labels:
            path = os.path.join(self.IMAGES_PATH, label)
            if not os.path.exists(path):
                os.mkdir(path)

        print("Workspace set up.")

    def capture_images(self):
        for label in self.labels:
            cam = cv2.VideoCapture(0)
            print('Collecting images for {}'.format(label))
            time.sleep(5)
            img_counter = 0
            while img_counter < self.number_imgs:
                ret, frame = cam.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                cv2.imshow("test", frame)
                k = cv2.waitKey(1)
                if k % 256 == 27:
                    # ESC pressed
                    print("Escape hit, closing")
                    break
                elif k % 256 == 32:
                    # SPACE Pressed
                    imgname = os.path.join(self.IMAGES_PATH, label, label + '.' + '{}.jpg'.format(str(uuid.uuid1())))
                    cv2.imwrite(imgname, frame)
                    print("{} written!".format(imgname))
                    img_counter += 1

        cam.release()
        cv2.destroyAllWindows()

    def label_images(self):
        os.system("python " + ".\\" + self.LABELIMG_PATH + "\\labelImg.py")


class TFOD:
    def __init__(self, labels, images_path):
        self.labels = labels
        self.IMAGES_PATH = images_path
        CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
        self.PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
        self.PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
        TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
        LABEL_MAP_NAME = 'label_map.pbtxt'

        self.paths = {
            'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
            'SCRIPTS_PATH': os.path.join('Tensorflow', 'scripts'),
            'APIMODEL_PATH': os.path.join('Tensorflow', 'models'),
            'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace', 'annotations'),
            'IMAGE_PATH': os.path.join('Tensorflow', 'workspace', 'images'),
            'MODEL_PATH': os.path.join('Tensorflow', 'workspace', 'models'),
            'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace', 'pre-trained-models'),
            'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME),
            'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'export'),
            'TFJS_PATH': os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'tfjsexport'),
            'TFLITE_PATH': os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'tfliteexport'),
            'PROTOC_PATH': os.path.join('Tensorflow', 'protoc')
        }

        self.files = {
            'PIPELINE_CONFIG': os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'pipeline.config'),
            'TF_RECORD_SCRIPT': os.path.join(self.paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME),
            'LABELMAP': os.path.join(self.paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
        }

    def check_paths(self):
        for path in self.paths.values():
            if not os.path.exists(path):
                if os.name == 'nt':
                    os.mkdir(path)

        wget.download(self.PRETRAINED_MODEL_URL)

    def create_label_maps(self):
        labels = []
        accum = 1
        for i in range(len(self.labels)):
            label_dict = {"name": "", "id": 0}
            label_dict["name"] = self.labels[i]
            label_dict["id"] = i + 1
            accum += 1
            labels.append(label_dict)
        # labels = [{'name': 'MiddleFinger', 'id': 1}]
        with open(self.files['LABELMAP'], 'w') as f:
            for label in labels:
                f.write('item { \n')
                f.write('\tname:\'{}\'\n'.format(label['name']))
                f.write('\tid:{}\n'.format(label['id']))
                f.write('}\n')

    def create_tf_records(self):
        os.system("python generate_tfrecord.py -x Tensorflow\\workspace\\images\\train -l "
                  "Tensorflow\\workspace\\annotations\\label_map.pbtxt -o "
                  "Tensorflow\\workspace\\annotations\\train.record")

        os.system("python generate_tfrecord.py -x Tensorflow\\workspace\\images\\test -l "
                  "Tensorflow\\workspace\\annotations\\label_map.pbtxt -o "
                  "Tensorflow\\workspace\\annotations\\test.record")

    def update_config_for_learning(self):

        # Make sure config is copied into Tensorflow/models/my_ssd folder
        config = config_util.get_configs_from_pipeline_file(self.files['PIPELINE_CONFIG'])
        pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
        with tf.io.gfile.GFile(self.files['PIPELINE_CONFIG'], "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, pipeline_config)

        pipeline_config.model.ssd.num_classes = len(self.labels)
        pipeline_config.train_config.batch_size = 4
        pipeline_config.train_config.fine_tune_checkpoint = os.path.join(self.paths['PRETRAINED_MODEL_PATH'],
                                                                         self.PRETRAINED_MODEL_NAME, 'checkpoint',
                                                                         'ckpt-0')
        pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
        pipeline_config.train_input_reader.label_map_path = self.files['LABELMAP']
        pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [
            os.path.join(self.paths['ANNOTATION_PATH'], 'train.record')]
        pipeline_config.eval_input_reader[0].label_map_path = self.files['LABELMAP']
        pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [
            os.path.join(self.paths['ANNOTATION_PATH'], 'test.record')]

        config_text = text_format.MessageToString(pipeline_config)
        with tf.io.gfile.GFile(self.files['PIPELINE_CONFIG'], "wb") as f:
            f.write(config_text)

    def train_tf_model(self):
        TRAINING_SCRIPT = os.path.join(self.paths['APIMODEL_PATH'], 'research', 'object_detection', 'model_main_tf2.py')
        command = "python {} --model_dir={} --pipeline_config_path={} --num_train_steps=2000".format(TRAINING_SCRIPT,
                                                                                                     self.paths[
                                                                                                         'CHECKPOINT_PATH'],
                                                                                                     self.files[
                                                                                                         'PIPELINE_CONFIG'])
        print(command)
        os.system(command)

    def evaluate_model(self):
        TRAINING_SCRIPT = os.path.join(self.paths['APIMODEL_PATH'], 'research', 'object_detection', 'model_main_tf2.py')
        command = "python {} --model_dir={} --pipeline_config_path={} --checkpoint_dir={}".format(TRAINING_SCRIPT,
                                                                                                  self.paths[
                                                                                                      'CHECKPOINT_PATH'],
                                                                                                  self.files[
                                                                                                      'PIPELINE_CONFIG'],
                                                                                                  self.paths[
                                                                                                      'CHECKPOINT_PATH'])
        print(command)
        os.system(command)
        # To see eval metrics go to eval/train folder and run tensorboard --logdir=. from the folder
        # and open your browser to the point.

    def detect_image(self):
        # Load pipeline config and build a detection model
        configs = config_util.get_configs_from_pipeline_file(self.files['PIPELINE_CONFIG'])

        detection_model = model_builder.build(model_config=configs['model'], is_training=False)
        # Restore checkpoint
        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore(os.path.join(self.paths['CHECKPOINT_PATH'], 'ckpt-5')).expect_partial()

        @tf.function
        def detect_fn(image):
            print("1")
            image, shapes = detection_model.preprocess(image)
            print("2")
            prediction_dict = detection_model.predict(image, shapes)
            print("3")
            detections = detection_model.postprocess(prediction_dict, shapes)
            print("4")
            return detections

        category_index = label_map_util.create_category_index_from_labelmap(self.files['LABELMAP'])
        IMAGE_PATH = os.path.join(self.paths['IMAGE_PATH'], 'test',
                                  'wirestripper.05923ad2-de13-11ec-9001-683e26c533e7.jpg')
        img = cv2.imread(IMAGE_PATH)
        image_np = np.array(img)

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections
        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'] + label_id_offset,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=5,
            min_score_thresh=.85,
            agnostic_mode=False)

        plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
        plt.show()

    def detect_real_time_obj(self):
        # Load pipeline config and build a detection model
        configs = config_util.get_configs_from_pipeline_file(self.files['PIPELINE_CONFIG'])

        detection_model = model_builder.build(model_config=configs['model'], is_training=False)
        # Restore checkpoint
        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore(os.path.join(self.paths['CHECKPOINT_PATH'], 'ckpt-3')).expect_partial()

        category_index = label_map_util.create_category_index_from_labelmap(self.files['LABELMAP'])
        cap = cv2.VideoCapture(0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        @tf.function
        def detect_fn(image):
            print("Creating predictions...")
            image, shapes = detection_model.preprocess(image)
            prediction_dict = detection_model.predict(image, shapes)
            detections = detection_model.postprocess(prediction_dict, shapes)
            return detections

        while cap.isOpened():

            ret, frame = cap.read()
            image_np = np.array(frame)
            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)

            detections = detect_fn(input_tensor)

            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                          for key, value in detections.items()}
            detections['num_detections'] = num_detections

            # detection_classes should be ints.
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

            label_id_offset = 1
            image_np_with_detections = image_np.copy()

            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes'] + label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.8,
                agnostic_mode=False)

            cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))

            if cv2.waitKey(10) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break

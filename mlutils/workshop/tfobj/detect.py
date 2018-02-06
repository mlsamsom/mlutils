import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from scipy.misc import imread
import cv2

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# sys.path.append('/home/mike/git_repos/models/research')

from utils import label_map_util

MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90


def maybe_download():
    if not os.path.exists(MODEL_NAME):
        opener = urllib.request.URLopener()
        opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
        tar_file = tarfile.open(MODEL_FILE)
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, os.getcwd())


class Detector(object):
    """Wraps a pretrained detector
    """
    def __init__(self, saveDir, labelMap, numClasses):
        self.saveDir = saveDir
        self.graph_file = os.path.join(saveDir, 'frozen_inference_graph.pb')
        self.detection_graph = tf.Graph()
        self._labelMap = label_map_util.load_labelmap(labelMap)
        self._cats = label_map_util.convert_label_map_to_categories(self._labelMap,
                                                                    max_num_classes=numClasses,
                                                                    use_display_name=True)
        self.catIndex = label_map_util.create_category_index(self._cats)

        print("[INFO] loading graph")
        self.load()

    def load(self):
        """Load the graph into the object
        """
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.graph_file, "rb") as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

    def predict(self, image, threshold=0.9):
        """Predict the bounding boxes

        Image is expected to be a numpy array
        """
        h, w = image.shape[:2]
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                # input and output tensors
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

                # detection scores represent confidences
                detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

                # load in image
                image_exp = np.expand_dims(image, axis=0)

                # run the detection
                boxes, scores, classes, num = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor:image_exp}
                )
                boxes = np.squeeze(boxes)
                classes = np.squeeze(classes)
                scores = np.squeeze(scores)

                detections = []
                for i in range(len(boxes)):
                    if scores[i] > threshold:
                        detection = {}
                        classname = self.catIndex[classes[i]]['name']
                        score = scores[i]

                        # re-scale box to the image
                        ymin, xmin, ymax, xmax = boxes[i]
                        ymin *= h
                        ymax *= h
                        xmin *= w
                        xmax *= w

                        # top, left, bottom, right
                        box = (int(ymin), int(xmin), int(ymax), int(xmax))

                        # pack
                        detection['label'] = classname
                        detection['box'] = box
                        detection['conf'] = score
                        detections.append(detection)

        return detections


if __name__ == '__main__':
    maybe_download()
    test_image = imread('test.jpg')
    detector = Detector(MODEL_NAME, PATH_TO_LABELS, NUM_CLASSES)
    dets = detector.predict(test_image)
    print(dets)

    font = cv2.FONT_HERSHEY_SIMPLEX
    for det in dets:
        y1, x1, y2, x2 = det['box']
        cv2.putText(test_image, det['label'], (x1+10, y1+30),
                    font, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(test_image, (x1, y1), (x2, y2), (255, 0, 0), 3)

    plt.imshow(test_image)
    plt.show()

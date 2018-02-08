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

from utils import label_map_util


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
    """
    mAP on coco dataset
    ssd mobilnet is fastest (30ms, mAP 21)
    ssd_inception is fast (42ms, mAP 24)
    frcnn resnet101 is twice as slow but accurate (106ms, mAP 32)
    frcnn inception resnet atrous is most accurate but very slow (620ms, 37mAP)

    coco dataset has 90, open image has 543
    """

    test_image = imread('/home/mike/Documents/tf_obj/test.jpg')

    modelPath = 'trained_networks/ssd_inception_v2_coco_2017_11_17'
    # modelPath = 'trained_networks/faster_rcnn_resnet101_coco_2017_11_08'
    # modelPath = 'trained_networks/faster_rcnn_inception_resnet_v2_atrous_coco_2017_11_08'
    # modelPath = 'trained_networks/ssd_mobilenet_v1_coco_2017_11_17'
    cocoLabels = 'data/mscoco_label_map.pbtxt'
    cocoClasses = 90

    detector = Detector(modelPath, cocoLabels, cocoClasses)
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

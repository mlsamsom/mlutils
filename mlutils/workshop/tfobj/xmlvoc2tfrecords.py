import xml.etree.ElementTree as ET
from PIL import Image
import tensorflow as tf

from utils import dataset_util


def voc2tfrecord(xml_file, output_file):
    """Convert data set with PASCAL VOC annotation xml to tfrecord

    Assumes that the image paths in the xml file exist

    Args:
        xml_file (str): Path to xml file
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    for image in root.iter('image'):
        image_file = image.get('file')



if __name__ == "__main__":
    fn = "/Users/mike/Documents/recognition_data/hotdogs.xml"
    voc2tfrecord(fn, 'test.tfrecord')

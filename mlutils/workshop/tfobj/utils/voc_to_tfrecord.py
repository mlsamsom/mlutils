import xml.etree.ElementTree as ET
import os


def xml2tfrecord(xml_file):
    """Convert PASCAL VOC xml to a TFRecord file
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()


if __name__ == "__main__":
    fn = "../dlib_rear_end_vehicles/testing.xml"

import xml.etree.ElementTree as ET


def voc2tfrecord(xml_file, output_file):
    """Convert data set with PASCAL VOC annotation xml to tfrecord

    Assumes that the image paths in the xml file exist

    Args:
        xml_file (str): Path to xml file
    """
    tree = ET.parse(xml_file)


if __name__ == "__main__":
    fn = "/Users/mike/Downloads/testing.xml"
    voc2tfrecord(fn, 'test.tfrecord')

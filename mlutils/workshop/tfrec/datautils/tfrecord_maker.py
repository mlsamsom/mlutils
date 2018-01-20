import os
import progressbar

import numpy as np
import tensorflow as tf
import skimage.io as io
from sklearn.preprocessing import LabelBinarizer

from pathutils import list_images


def _byte_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_list(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def write_tfrecord_recognition(trainDir, outFile):
    """Convert a raw training set into tfrecords file

    Stores a recognition training set
    Assumes set is set up like /path/to/{label}/image.jpg

    Args:
        trainDir (str): Directory containing training set
        outFile (str): File path to save data
    """
    # get the list of input images
    writer = tf.python_io.TFRecordWriter(outFile)

    # fit the label binarizer
    imagePaths = list(list_images(trainDir))
    classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
    labelBin = LabelBinarizer()
    labelBin.fit(classNames)

    w = [' ', progressbar.Timer(), ' ', progressbar.Bar(), ' ', progressbar.ETA(), ' ']
    bar = progressbar.ProgressBar(widgets=w, max_value=len(imagePaths))
    for i, imagePath in enumerate(imagePaths):
        img = io.imread(imagePath)
        label = imagePath.split(os.path.sep)[-2]

        # store image size to read serialized images later
        height, width = img.shape[:2]

        # convert image to string
        img_raw = img.tostring()

        encoded_label = labelBin.transform([label])[0]

        # create storage dictionary and add to record
        recDict = {'height': _int64_feature(height),
                   'width': _int64_feature(width),
                   'image_raw': _byte_feature(img_raw),
                   'label': _int64_list(encoded_label)}

        tfrec = tf.train.Example(features=tf.train.Features(feature=recDict))

        writer.write(tfrec.SerializeToString())
        bar.update(i)

    bar.finish()
    writer.close()


if __name__ =='__main__':
    test_dir = '/Users/mike/Pictures/Mercedes_Images'
    output_dir = '/Users/mike/Pictures/mercedes.tfrecords'
    write_tfrecord_recognition(test_dir, output_dir)

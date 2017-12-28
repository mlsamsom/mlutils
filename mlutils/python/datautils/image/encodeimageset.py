from skimage.io import imread
import progressbar
import uuid
import os
import numpy as np
import base64
import json


def base64_encode_array(inArray):
    """Return base64 encoded array

    Args:
        inArray (ndarray) : Image array

    Returns:
        str : Image encoded in a base64 string
    """
    return base64.b64encode(inArray)


def base64_decode_array(inStr, dtype):
    """Decodes and encoded array

    Args:
        inStr (str): a base64 encoded image
        dtype (str): The data type of the image string

    Returns:
        ndarray: An image array
    """
    return np.frombuffer(base64.decodestring(inStr), dtype=dtype)


def base64_encode_image(inArray):
    """Converts and array to JSON encoded list

    The return list includes image data, image type
    and image shape.

    Args:
        inArray (ndarray) : An array to encode

    Returns:
        str : A JSON string with encoded image data
    """
    imgDat = [base64_encode_array(inArray).decode("utf-8")]
    imgType = str(inArray.dtype)
    imgShape = inArray.shape
    return json.dumps([ imgDat, imgType, imgShape ])


def base64_decode_image(inStr):
    """Decodes a JSON encoded image

    The JSON string should include image data,
    image type and image shape.

    Args:
        inStr (str): JSON encoded image

    Returns:
        ndarray: an image array
    """
    imgDat, imgType, imgShape = json.loads(inStr)
    imgDat = bytes(imgDat, encoding="utf-8")

    imgDat = base64_decode_array(imgDat, imgType)
    imgDat = imgDat.reshape(imgShape)
    return imgDat


def hdfsEncodeBatch(parentDir, output):
    """Encodes images in a training set for HDFS recognition

    Images must be organized in directories with the label of
    the directory as the image labels

    Args:
        parentDir (str): parent directory containing images
        output (str): path to output file

    Returns:
        dict: A count of labels in the dataset
    """
    labels = [x for x in os.listdir(parentDir)
              if os.path.isdir(os.path.join(parentDir, x))]

    f = open(output, "w")

    summary = {}
    for label in labels:
        # get images in label directory
        labelPath = os.path.join(parentDir, label)
        imageNames = [x for x in os.listdir(labelPath)
                      if 'jpg' in x]

        # set up bar name
        barName = "Serializing {}".format(label)
        widgets = [barName, progressbar.Percentage(),
                   " ", progressbar.Bar(),
                   " ", progressbar.ETA()]
        bar = progressbar.ProgressBar(widgets=widgets,
                                      max_value=len(imageNames))

        for i, imageName in bar(enumerate(imageNames)):
            imagePath = os.path.join(labelPath, imageName)
            imageID = str(uuid.uuid4())
            image = base64_encode_image(imread(imagePath))
            f.write('{}\t{}\t{}\t{}\n'.format(imageID, label, imagePath, image))
        summary[label] = i
    f.close()
    return summary


class HDF5EncoderBase(object):
    """Base encoder for hdf5 data

    The transform method must be overridden for your own datasets
    """
    def __init__(self, dims):
        self.dims = dims

    def transform(self):
        pass


if __name__ == '__main__':
    import argparse
    from argparse import ArgumentParser

    default_path = '/Users/mike/Documents/recognition_data'
    default_output = '/Users/mike/GitRepos/mlutils/data/recognition.txt'
    ap = ArgumentParser()
    ap.add_argument('-d', '--directory', default=default_path,
                    help='Path to directory')
    ap.add_argument('-s','--save', default=default_output)
    args = ap.parse_args()

    print(hdfsEncodeBatch(args.directory, args.save))


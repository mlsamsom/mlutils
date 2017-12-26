from skimage.io import imread
import progressbar
import uuid
import os
import time

import encodings


def encodeBatch(parentDir):
    """Encodes images in a training set for recognition

    Images must be organized in directories with the label of
    the directory as the image labels

    Args:
        parentDir (str): parent directory containing images

    Returns:

    """
    labels = [x for x in os.listdir(parentDir)
              if os.path.isdir(os.path.join(parentDir, x))]

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
            time.sleep(0.1)


if __name__ == '__main__':
    import argparse
    from argparse import ArgumentParser

    default_path = '/Users/mike/Documents/recognition_data'
    ap = ArgumentParser()
    ap.add_argument('-d', '--directory', default=default_path,
                    help='Path to directory')
    ap.add_argument('-s','--save', default='../data/recognition')
    args = ap.parse_args()

    encodeBatch(args.directory)



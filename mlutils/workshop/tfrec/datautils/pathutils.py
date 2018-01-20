import os


def list_files(basePath, validExts=('.jpg', '.jpeg', '.png', '.bmp'), contains=None):
    """Create a list of files in a nested directory
    """
    # loop over the directory structure
    for rootDir, dirNames, filenames in os.walk(basePath):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue
            # ignore dotfiles
            if filename.startswith('.'):
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()

            # check to see if the file is an image and should be processed
            if ext.endswith(validExts):
                # construct the path to the image and yield it
                imagePath = os.path.join(rootDir, filename).replace(" ", "\\ ")
                yield imagePath


def list_images(basePath, contains=None):
    """Create a list of images in a dataset
    """
    l = list_files(basePath,
                   validExts=('.jpg', '.jpeg', '.png', '.bmp'),
                   contains=contains)
    return l

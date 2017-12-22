# USAGE
# python download_images.py --urls urls.txt --output images/santa

# import the necessary packages
import requests
import cv2
import os
try:
    import Tkinter
    import tkFileDialog
except ImportError:
    import tkinter as Tkinter
    import tkinter.filedialog as tkFileDialog


def guiGetFilePath(msg):
    root = Tkinter.Tk()
    root.withdraw()

    curdir = os.getcwd()
    directory = tkFileDialog.askopenfilename(
        parent=root, initialdir=curdir, title=msg)
    return directory


def download(urlfile, outfile):
    # grab the list of URLs from the input file, then initialize the
    # total number of images downloaded thus far
    with open(urlfile, 'r') as urlf:
        urls = urlf.read().strip().split("\n")

    print("[INFO] Downloading Images")
    total = 0
    for url in urls:
        try:
            # try to download the image
            r = requests.get(url, timeout=60)

            # save the image to disk
            imgnm = "{}.jpg".format(str(total).zfill(8))
            p = os.path.join(outfile, imgnm)
            with open(p, "wb") as f:
                f.write(r.content)

            # update the counter
            print("[INFO] downloaded: {}".format(p))
            total += 1

        # handle if any exceptions are thrown during the download process
        except:
            print("[INFO] error downloading {}...skipping".format(p))

    # loop over the image paths we just downloaded
    for image in os.listdir(outfile):
        imagePath = os.path.join(outfile, image)

        # initialize if the image should be deleted or not
        delete = False

        # try to load the image
        try:
            image = cv2.imread(imagePath)

            # if the image is `None` then we could not properly load it
            # from disk, so delete it
            if image is None:
                print("None")
                delete = True

        # if OpenCV cannot load the image then the image is likely
        # corrupt so we should delete it
        except:
            print("Except")
            delete = True

        # check to see if the image should be deleted
        if delete:
            print("[INFO] deleting {}".format(imagePath))
            os.remove(imagePath)


if __name__ == "__main__":
        urlFile = guiGetFilePath("Select Url File")
        urlName = os.path.basename(urlFile).split(".")[0]
        urlFolder = os.path.dirname(urlFile)
        outPath = os.path.join(urlFolder, urlName)
        if not os.path.exists(outPath):
            os.mkdir(outPath)

        download(urlFile, outPath)

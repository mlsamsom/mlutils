"""Tools to encode images for storage

This module is taken from Adrian Rosebrocks imutils
library to to minimize dependencies.
"""

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
    return json.dumps(imgDat, imgType, imgShape)


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

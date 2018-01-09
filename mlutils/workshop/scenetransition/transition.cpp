#include "transition.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdexcept>

using namespace cv;
using namespace std;

namespace transdet
{

  // HELPERS
  //------------------------------------------------------------------------------------

  double _medianChannel(const cv::Mat &input)
  {
    // compute histogram on a channel in an image
    int nVals = 256;
    const int* channels = {0};
    float range[] = { 0, (float)nVals };
    const float* histRange = { range };
    bool uniform = true; bool accumulate = false;
    cv::Mat hist;
    cv::calcHist(&input, 1, channels, cv::Mat(), hist, 1, &nVals, &histRange, uniform, accumulate);

    // calculate CDF
    cv::Mat cdf;
    hist.copyTo(cdf);
    for (int i = 1; i < nVals; i++)
      {
        cdf.at<float>(i) += cdf.at<float>(i - 1);
      }
    cdf /= input.total();

    // compute median
    double medianVal;
    for (int i = 0; i < nVals; i++)
      {
        if (cdf.at<float>(i) >= 0.5) {
          medianVal = i;
          break;
        }
      }
    return (double)medianVal;
  }

  //------------------------------------------------------------------------------------

  double _medianMat(const cv::Mat &input)
  {
    // Grayscale image
    if (input.channels() == 1) {
      return _medianChannel(input);

    // 3 channel image
    }else if (input.channels() == 3) {
      // split image
      std::vector<cv::Mat> bgr;
      cv::split(input, bgr);

      cv::Mat flat;
      cv::hconcat(bgr, flat);
      return _medianChannel(flat);

    // ignoring 4 channel images
    } else {
      throw std::length_error("Invalid image channels");
    }
  }

  //------------------------------------------------------------------------------------

  double _frameDiff(const cv::Mat &canny1, const cv::Mat &canny2, const int &numIter)
  {
    double percent_diff = 1.0;
    Mat e_1;
    Mat e_2;

    // dilate images
    Mat dilation_elem = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    dilate(canny1, e_1, dilation_elem, cv::Point(-1, -1), numIter);
    dilate(canny2, e_2, dilation_elem, cv::Point(-1, -1), numIter);

    // calculate percent differences
    Mat tmpand;
    cv::bitwise_and(e_1, e_2, tmpand);
    percent_diff = cv::sum(tmpand)[0] / cv::sum(e_1)[0];

    return percent_diff;
  }

  //------------------------------------------------------------------------------------

  void _customCanny(const cv::Mat &src, cv::Mat &dst)
  {
    // find thresholds
    double median = _medianMat(src);
    double sigma = 0.33;

    cout << median << endl;

    int lowThresh = (int)std::max(0.0, (1.0 - sigma) * median);
    int highThresh = (int)std::min(0.0, (1.0 - sigma) * median);

    cout << lowThresh << endl;
    cout << highThresh << endl;

    // compute canny
    Canny(src, dst, lowThresh, highThresh, 4);
  }

  //------------------------------------------------------------------------------------

  double _hammingDist(const cv::Mat &binImg1,
                      const cv::Mat &binImg2,
                      const int &xShift,
                      const int &yShift)
  {
    double dist = -1.0;
    // loop through the matrix, calculate the mate and then distance
    if ((binImg1.rows != binImg2.rows) || (binImg1.cols != binImg2.cols)) {
      cout << "[ERROR] in _hammingDist" << endl;
      cout << "Images must be the same size" << endl;
      return dist;
    }

    int xrange = binImg1.cols;
    int yrange = binImg1.rows;
    double total = 0.0;
    // TODO optimize loop
    // TODO batch the rows into chars or something to speed up bitwise operation
    // TODO or change the implementation to use a border of radius and opencv bitwise_xor
    uint8_t* pixelPtr1 = (uint8_t*)binImg1.data;
    uint8_t* pixelPtr2 = (uint8_t*)binImg2.data;
    for (int row; row < yrange; row++)
      {
        for (int col; col < xrange; col++)
          {
            // bitwise hamming distance
            int rhscol;
            if (rhscol > xrange) {
              rhscol = xrange - xShift;
            } else {
              rhscol = col + xShift;
            }

            int rhsrow;
            if (rhsrow > yrange) {
              rhsrow = yrange - yShift;
            } else {
              rhsrow = row + yShift;
            }

            uint8_t bin1Pix = pixelPtr1[row*xrange + col];
            uint8_t bin2Pix = pixelPtr2[rhsrow*xrange + rhscol];
            if (bin1Pix ^ bin2Pix) { total += 1.0; }
          }
      }
    dist = total / ( (double)xrange * (double)yrange );
    return dist;
  }

  //------------------------------------------------------------------------------------

  double _hausdorffDist(cv::Mat &binImg1, cv::Mat &binImg2, int &xShift, int &yShift)
  {
    cout << "[ERROR] _hausdorffDist not implemented" << endl;
    return -1.0;
  }

  //------------------------------------------------------------------------------------

  // MAIN FUNCTIONS
  //------------------------------------------------------------------------------------
  void rollCvMat(cv::Mat &dst,
                 const int &shift,
                 const int &axis)
  {
    // TODO figure out why double roll is broken
    // making a copy for now
    const Mat orig = dst.clone();

    const int height = orig.rows;
    const int width = orig.cols;

    if (axis == 0) {
      const int shiftPoint = height-shift;

      // roll rows
      // check shift inputs
      if (shift < 0 || shift > height) {
        throw std::invalid_argument("Recieved an invalid shift");
      }

      // if no shift just copy the image into the dst
      if (shift == 0 || shift == height) {
        dst == orig;
        return;
      }

      // perform roll op
      cv::vconcat(orig(cv::Rect(0, shiftPoint, width, shift)),
                  orig(cv::Rect(0, 0, width, shiftPoint)),
                  dst);

    }else if (axis == 1) {
      const int shiftPoint = width-shift;

      // roll cols
      if (shift < 0 || shift > orig.cols) {
        throw std::invalid_argument("Recieved an invalid shift");
      }

      // if no shift just copy the image into the dst
      if (shift == 0 || shift == orig.cols) {
        dst = orig;
        return;
      }

      // perform roll op
      cv::hconcat(orig(cv::Rect(shiftPoint, 0, shift, height)),
                  orig(cv::Rect(0, 0, shiftPoint, height)),
                  dst);

    } else {
      throw std::invalid_argument("Recieved and invalid axis values");
    }

  }

  cv::Point globalEdgeMotion(const cv::Mat &canny1,
                             const cv::Mat &canny2,
                             const int &radius)
  {
    std::vector<double> distances;
    std::vector<cv::Point> displacements;

    // get hamming distances
    for (int dx = -radius; dx <= radius; dx++)
      {
        for (int dy = -radius; dy <= radius; dy++)
          {
            // calculate the distance between canny1 and canny2 pixels
            // within dx, dy offset
            distances.push_back(_hammingDist(canny1, canny2, dx, dy));
            displacements.push_back(cv::Point(dx, dy));
          }
      }

    // get smallest displacement
    std::vector<double>::iterator result = std::min_element(std::begin(distances),
                                                            std::end(distances));
    int idx = std::distance(std::begin(distances), result);
    return displacements[idx];
  }

  //------------------------------------------------------------------------------------

  std::vector<int> sceneDetEdges(const std::vector<cv::Mat> &vidArray,
                                 const float &threshold,
                                 const int &minSceneLen)
  {
    // The first frame is always the beginning of the scene
    std::vector<int> detectedScene = {0};
    int numIter = 6;

    for (int i = 0; i < vidArray.size() - 1; i++)
      {
        Mat grayNow, grayNext, cannyNow, cannyNext;

        // convert to grayscale
        cvtColor(vidArray[i], grayNow, cv::COLOR_RGB2GRAY);
        cvtColor(vidArray[i+1], grayNext, cv::COLOR_RGB2GRAY);

        // get canny transforms for this and the next frames
        // no need to reduce noise for this application
        _customCanny(grayNow, cannyNow);
        _customCanny(grayNext, cannyNext);

        // calculate global edge motion between cannyNow and cannyNext
        cv::Point motion = globalEdgeMotion(cannyNow, cannyNext, 6);

        // compute the percent difference
        // TODO implement roll
        // rollCvMat(cannyNow, motion.y, 0);
        // rollCvMat(cannyNow, motion.x, 1);

        // if the difference is over the threshold we found a scene transition
      }
  }

  //------------------------------------------------------------------------------------

} // namespace transdet

int main()
{
  cout << "RUNNING TESTS" << endl;

  Mat image1;
  image1 = imread( "../testims/mt1.jpg", 1 );
  if ( !image1.data ) {
    printf("No image data \n");
    return -1;
  }

  Mat image2;
  image2 = imread( "../testims/mt2.jpg", 1 );
  if ( !image2.data ) {
    printf("No image data \n");
    return -1;
  }

  //------------------------------------------------------------------------------------
  cout << "\nTesting _medianMat" << endl;
  double med = transdet::_medianMat(image1);
  if (med == 121.0) {
    cout << "[SUCCESS] _medianMat" << endl;
  } else {
    cout << "[FAILED] _medianMat" << endl;
  }

  //------------------------------------------------------------------------------------
  cout << "\nTesting _frameDiff" << endl;
  // resize image 2 to equal image 1


  // get grayscales

  // get cannys

  // get percent difference
  // should be around 0.00044083595275878906

  //------------------------------------------------------------------------------------

  // //------------------------------------------------------------------------------------
  // cout << "Testing roll" << endl;
  // transdet::rollCvMat(image1, 50, 1);
  // transdet::rollCvMat(image1, 50, 0);

  // namedWindow("Display Image", WINDOW_AUTOSIZE );
  // imshow("Display Image", image1);
  // waitKey(0);

  // //------------------------------------------------------------------------------------
  // cout << "Testing globalEdgeMotion" << endl;
  // Mat grayNow, grayNext, cannyNow, cannyNext;

  // // resize second image to match first
  // cv::Mat image2_rs;
  // cv::resize(image2, image2_rs, image1.size());

  // // convert to grayscale
  // cvtColor(image1, grayNow, cv::COLOR_RGB2GRAY);
  // cvtColor(image2_rs, grayNext, cv::COLOR_RGB2GRAY);

  // // get canny transforms for this and the next frames
  // // no need to reduce noise for this application
  // transdet::_customCanny(grayNow, cannyNow);
  // transdet::_customCanny(grayNext, cannyNext);

  // // calculate global edge motion between cannyNow and cannyNext
  // cv::Point motion = transdet::globalEdgeMotion(cannyNow, cannyNext, 6);

  // //------------------------------------------------------------------------------------
  // cout << "Testing sceneDetEdges" << endl;

  // //------------------------------------------------------------------------------------
  // cout << "Testing sceneDetColors" << endl;

  return 0;
}

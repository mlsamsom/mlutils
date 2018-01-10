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
    cv::calcHist(&input, 1, channels, cv::Mat(), hist, 1, &nVals,
                 &histRange, uniform, accumulate);

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

    return 1.0 - percent_diff;
  }

  //------------------------------------------------------------------------------------

  void _customCanny(const cv::Mat &src, cv::Mat &dst)
  {
    // find thresholds
    double median = _medianMat(src);
    double sigma = 0.33;

    int lowThresh = (int)std::max(0.0, (1.0 - sigma) * median);
    int highThresh = (int)std::min(255.0, (1.0 + sigma) * median);

    // compute canny
    Canny(src, dst, lowThresh, highThresh, 3);
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
      throw std::out_of_range("_hammingDist");
    }

    // TODO use C style pointer access instead of the copt to improve efficiency
    Mat next = binImg2.clone();
    rollCvMat(next, yShift, 0);
    rollCvMat(next, xShift, 1);

    Mat xorImg;
    bitwise_xor(binImg1, next, xorImg);
    auto total = cv::sum(xorImg)[0];

    dist = (double)total / ( 255 * (double)binImg1.cols * (double)binImg2.rows );
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
    // NOTE the better way to do this might be to rearrange the Mat header
    // might be able to hack it with std::rotate?
    const Mat orig = dst.clone();

    const int height = orig.rows;
    const int width = orig.cols;

    if (axis == 0) {
      // roll rows
      // check shift inputs
      if (shift > height) {
        throw std::invalid_argument("Recieved an invalid shift");
      }

      // if no shift just copy the image into the dst
      if (shift == 0 || shift == height) {
        dst = orig;
        return;
      }

      if (shift > 0) {
        const int shiftPoint = height-shift;
        cv::vconcat(orig(cv::Rect(0, shiftPoint, width, shift)),
                    orig(cv::Rect(0, 0, width, shiftPoint)),
                    dst);

      }else if (shift < 0) {
        int pshift = -shift;
        cv::vconcat(orig(cv::Rect(0, pshift, width, height-pshift)),
                    orig(cv::Rect(0, 0, width, pshift)),
                    dst);
      } else {
        cout << "impossible" << endl;
      }

    }else if (axis == 1) {
      // roll cols
      if (shift > orig.cols) {
        throw std::invalid_argument("Recieved an invalid shift");
      }

      // if no shift just copy the image into the dst
      if (shift == 0 || shift == orig.cols) {
        dst = orig;
        return;
      }

      // perform roll op
      if (shift > 0) {
        const int shiftPoint = width-shift;
        cv::hconcat(orig(cv::Rect(shiftPoint, 0, shift, height)),
                    orig(cv::Rect(0, 0, shiftPoint, height)),
                    dst);
      }else if (shift < 0) {
        int pshift = -shift;
        cv::hconcat(orig(cv::Rect(pshift, 0, width-pshift, height)),
                    orig(cv::Rect(0, 0, pshift, height)),
                    dst);
      }

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
            auto distance = _hammingDist(canny1, canny2, dx, dy);
            distances.push_back(distance);
            cout << distance << endl;
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
    return detectedScene;
  }

  //------------------------------------------------------------------------------------

} // namespace transdet

int main()
{
  cout << "RUNNING TESTS" << endl << endl;

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
  double med = transdet::_medianMat(image1);
  if (med == 121.0) {
    cout << "[SUCCESS] _medianMat" << endl;
  } else {
    cout << "[FAILED] _medianMat" << endl;
  }

  //------------------------------------------------------------------------------------
  // cout << "\nTesting rollCvMat" << endl;
  // Mat posroll = image1.clone();
  // transdet::rollCvMat(posroll, 50, 0);
  // transdet::rollCvMat(posroll, 50, 1);
  // namedWindow("win", WINDOW_AUTOSIZE);
  // imshow("win", posroll);
  // waitKey(0);

  // Mat negroll = image1.clone();
  // transdet::rollCvMat(negroll, -50, 0);
  // transdet::rollCvMat(negroll, -50, 1);
  // namedWindow("win", WINDOW_AUTOSIZE);
  // imshow("win", negroll);
  // waitKey(0);

  //------------------------------------------------------------------------------------
  // resize image 2 to equal image 1
  Mat image2rz;
  cv::resize(image2, image2rz, image1.size());

  // get grayscales
  Mat gray1, gray2;
  cv::cvtColor(image1, gray1, cv::COLOR_BGR2GRAY);
  cv::cvtColor(image2rz, gray2, cv::COLOR_BGR2GRAY);

  // get Canny
  Mat canny1, canny2;
  transdet::_customCanny(gray1, canny1);
  transdet::_customCanny(gray2, canny2);

  // get percent difference
  double diff1 = transdet::_frameDiff(canny1, canny2, 1);
  int test = (int)(diff1*100);
  if (test == 27) {
    cout << "[SUCCESS] _frameDiff" << endl;
  } else {
    cout << "[FALIED] _frameDiff" << endl;
    cout << test << endl;
  }

  //------------------------------------------------------------------------------------
  // TODO make a better test
  Mat E = Mat::eye(10, 10, CV_8UC1);
  Mat O = Mat::ones(10, 10, CV_8UC1);
  Mat cE, cO;
  transdet::_customCanny(E, cE);
  transdet::_customCanny(O, cO);
  // cout << "E = " << endl << " " << E << endl << endl;

  // namedWindow("Dimg", WINDOW_AUTOSIZE);
  // imshow("Dimg", E);
  // waitKey(0);
  // namedWindow("Dimg", WINDOW_AUTOSIZE);
  // imshow("Dimg", O);
  // waitKey(0);

  // no shift
  double d1 = transdet::_hammingDist(cE, cO, 0, 0);
  // neg shift
  double d2 = transdet::_hammingDist(cE, cO, -1, -2);
  // pos shift
  double d3 = transdet::_hammingDist(cE, cO, 1, 2);

  if (d1 == 0.2 && d2 == 0.2 && d3 == 0.2) {
    cout << "[SUCCESS] _hammingDist" << endl;
  } else {
    cout << "[FAILED] _hammingDist" << endl;
    cout << d1 << " " << d2 << " " << d3 << endl;

    cout << "cE = " << endl << " " << cE << endl << endl;
    cout << "cO = " << endl << " " << cO << endl << endl;
  }

  //------------------------------------------------------------------------------------

  cv::Point motion = transdet::globalEdgeMotion(cE, cO, 6);

  cout << "cE: " << endl;
  cout << cE << endl;

  if (motion.x == 0 && motion.y == 0) {
    cout << "[SUCCESS] globalEdgeMotion" << endl;
  } else {
    cout << "[FAILED] globalEdgeMotion" << endl;
    cout << "motion: [" << motion.x << ", " << motion.y << "]\n";
    cout << "cE = " << endl << " " << cE << endl << endl;
    transdet::rollCvMat(cE, 3, 0);
    cout << "cE = " << endl << " " << cE << endl << endl;
  }

  // //------------------------------------------------------------------------------------
  // cout << "Testing sceneDetEdges" << endl;

  // //------------------------------------------------------------------------------------
  // cout << "Testing sceneDetColors" << endl;

  return 0;
}

#include "transition.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

namespace transdet
{

  // HELPERS
  //------------------------------------------------------------------------------------

  double _medianMat(cv::Mat &input, int &nVals)
  {
    // compute histogram
    float range[] = { 0, (float)nVals };
    const float* histRange = { range };
    int channels[] = {0};
    bool uniform = true;
    bool accumulate = false;
    int histSize[] = { nVals };

    cv::Mat hist;
    cv::calcHist(&input, 1, channels, cv::Mat(), hist, 1, histSize, &histRange);

    // calculate CDF
    cv::Mat cdf;
    hist.copyTo(cdf);
    for (int i = 1; i <= nVals-1; i++)
      {
        cdf.at<float>(i) += cdf.at<float>(i - 1);
      }
    cdf /= input.total();

    // compute median
    double medianVal;
    for (int i = 0; i <= nVals-1; i++)
      {
        if (cdf.at<float>(i) >= 0.5) {
          medianVal = i;
          break;
        }
      }
    return medianVal/nVals;
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
    double median = _medianMat(src, 2^8);
    double sigma 0.33;

    int lowThresh = (int)std::max(0, (1.0 - sigma) * median);
    int highThresh = (int)std::min(0, (1.0 - sigma) * median);

    // compute canny
    Canny(src, dst, lowThresh, highThresh, 4);
  }

  //------------------------------------------------------------------------------------

  double _hammingDist(cv::Mat &binImg1, cv::Mat &binImg2, int &xShift, int &yShift)
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
    for (int row; row < yrange; row++)
      {
        for (int col; col < xrange; col++)
          {
            // bitwise hamming distance
            rhscol = col + xShift;
            if (rhscol > xrange) {
              rhscol = xrange - xShift;
            }
            rhsrow = row + yShift;
            if (rhsrow > yrange) {
              rhsrow = yrange - yShift;
            }

            if (binImg1[row][col] ^ binImg2[rhsrow][rhscol]) { total += 1.0; }
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
  void rollCvMat(cv::Mat &m, const int &shift, const int &axis)
  {
    // re-implement numpy roll function
  }

  cv::Point globalEdgeMotion(const cv::Mat &canny1, const cv::Mat &canny2, const int &radius)
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
            distances.push_back(_hammingDistance(canny1, canny2, dx, dy));
            displacements.push_back(cv::Point(dx, dy))
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
        cv::Point motion = globalEdgeMotion(cannyNow, cannyNext);

        // compute the percent difference
        // TODO implement roll
        rollCvMat(cannyNow, motion.y, 0);
        rollCvMat(cannyNow, motion.x, 1);

        // if the difference is over the threshold we found a scene transition
      }
  }

  //------------------------------------------------------------------------------------

} // namespace transdet

int main(int argc, char** argv )
{
  cout << "RUNNING TEST" << endl;
  if ( argc != 3 ) {
    printf("usage: DisplayImage.out <Image_Path1> <Image_Path2\n");
    return -1;
  }

  Mat image1;
  image1 = imread( argv[1], 1 );
  if ( !image1.data ) {
    printf("No image data \n");
    return -1;
  }

  Mat image2;
  image2 = imread( argv[2], 1 );
  if ( !image2.data ) {
    printf("No image data \n");
    return -1;
  }

  namedWindow("Display Image", WINDOW_AUTOSIZE );
  imshow("Display Image", image1);
  waitKey(0);
  return 0;
}

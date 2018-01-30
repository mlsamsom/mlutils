#include "transition.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdexcept>

using namespace cv;
using namespace std;

namespace transdet
{

  //------------------------------------------------------------------------------------
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
                      const cv::Mat &binImg2)
  {
    double dist = -1.0;
    // loop through the matrix, calculate the mate and then distance
    if ((binImg1.rows != binImg2.rows) || (binImg1.cols != binImg2.cols)) {
      cout << "[ERROR] in _hammingDist" << endl;
      cout << "Images must be the same size" << endl;
      throw std::out_of_range("_hammingDist");
    }

    Mat xorImg;
    bitwise_xor(binImg1, binImg2, xorImg);
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

  double _frameDiffEdge(const cv::Mat &currentImage,
                        const cv::Mat &nextImage,
                        const int &motionIter,
                        const int &diffRadius)
  {
    Mat grayNow, grayNext, cannyNow, cannyNext;

    // convert to grayscale
    cvtColor(currentImage, grayNow, cv::COLOR_RGB2GRAY);
    cvtColor(nextImage, grayNext, cv::COLOR_RGB2GRAY);

    // get canny transforms for this and the next frames
    // no need to reduce noise for this application
    _customCanny(grayNow, cannyNow);
    _customCanny(grayNext, cannyNext);

    // calculate global edge motion between cannyNow and cannyNext
    cv::Point motion = globalEdgeMotion(cannyNow, cannyNext, motionIter);
    cannyNext = rollCvMat(cannyNext, motion.y, 0);
    cannyNext = rollCvMat(cannyNext, motion.x, 1);

    // compute the percent difference
    double p_in = _frameDiff(cannyNow, cannyNext, diffRadius);
    double p_out = _frameDiff(cannyNext, cannyNow, diffRadius);
    double p = std::max(p_in, p_out);

    return p;
  }

  //------------------------------------------------------------------------------------

  // MAIN FUNCTIONS
  //------------------------------------------------------------------------------------
  cv::Mat rollCvMat(const cv::Mat &a,
                    const int &shift,
                    const int &axis)
  {
    if (a.channels() != 1) {
      throw std::out_of_range("[ERROR] rollCvMat, must be 1 channel image");
    }

    if (shift == 0) {
      return a;
    }

    cv::Mat res(a.size(), a.type());

    const int height = a.rows;
    const int width = a.cols;
    int pshift;
    std::vector<int> idxs;

    if (axis == 0) {
      for (int i = 0; i < height; i++) idxs.push_back(i);

      if (shift < 0) {
        pshift = height + shift;
      } else {
        pshift = shift;
      }
      pshift = pshift % height;

      // NOTE could just use idxs to rearrange header?
      std::rotate(idxs.rbegin(), idxs.rbegin()+pshift, idxs.rend());

      for (int i = 0; i < height; i++)
        {
          a.row(idxs[i]).copyTo(res.row(i));
        }

    }else if (axis == 1) {
      for (int i = 0; i < width; i++) idxs.push_back(i);

      if (shift < 0) {
        pshift = width + shift;
      } else {
        pshift = shift;
      }

      pshift = pshift % width;
      std::rotate(idxs.rbegin(), idxs.rbegin()+pshift, idxs.rend());

      for (int i = 0; i < width; i++)
        {
          a.col(idxs[i]).copyTo(res.col(i));
        }
    } else {
      throw std::invalid_argument("Recieved and invalid axis values");
    }

    return res;
  }

  //------------------------------------------------------------------------------------

  cv::Point globalEdgeMotion(const cv::Mat &canny1,
                             const cv::Mat &canny2,
                             const int &radius)
  {
    std::vector<double> distances;
    std::vector<cv::Point> displacements;

    Mat cimage;
    // get hamming distances
    for (int dx = -radius; dx <= radius; dx++)
      {
        for (int dy = -radius; dy <= radius; dy++)
          {
            // calculate the distance between canny1 and canny2 pixels
            // within dx, dy offset
            cimage = rollCvMat(canny2, dy, 0);
            cimage = rollCvMat(cimage, dx, 1);

            auto distance = _hammingDist(cimage, canny1);
            distances.push_back(distance);
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
    const int motionIter = 6;
    const int diffRadius = 6;

    for (int i = 0; i < vidArray.size() - 1; i++)
      {
        double p = _frameDiffEdge(vidArray[i],
                                  vidArray[i+1],
                                  motionIter,
                                  diffRadius);

        // if the difference is over the threshold we found a scene transition
        bool lessThanThresh = p > threshold;
        bool sceneLenOK = (i - detectedScene[detectedScene.size()-1]) > minSceneLen;
        bool isTransition = lessThanThresh && sceneLenOK;

        if (isTransition) { detectedScene.push_back(i); }
      }
    return detectedScene;
  }

  //------------------------------------------------------------------------------------

  std::vector<int> sceneDetEdges(const std::string &vidPath,
                                 const float &threshold,
                                 const int &minSceneLen,
                                 const int &imageHeight,
                                 const int &imageWidth)
  {
    // clear the vector buffer

    const int motionIter = 6;
    const int diffRadius = 6;

    cv::VideoCapture cap(vidPath);
    if (!cap.isOpened()) {
      throw std::runtime_error("[ERROR] unable to open video");
    }

    std::vector<int> detectedScene = {0};
    FrameBuffer frameBuffer(2);
    Size sampleSize(imageWidth, imageHeight);

    int i = 0;
    for (;;)
      {
        Mat frame;
        cap >> frame;
        if (frame.empty()) { break; }

        // Downsample image
        cv::resize(frame, frame, sampleSize);

        frameBuffer.add(frame);

        if (frameBuffer.full()) {
          double p = _frameDiffEdge(frameBuffer.frames[0],
                             frameBuffer.frames[1],
                             motionIter,
                             diffRadius);

          bool lessThanThresh = p > threshold;
          bool sceneLenOK = (i - detectedScene[detectedScene.size()-1]) > minSceneLen;
          bool isTransition = lessThanThresh && sceneLenOK;

          if (isTransition) { detectedScene.push_back(i); }
        }
        i++;
      }
    cap.release();

    return detectedScene;
  }

  //------------------------------------------------------------------------------------

  // MAIN CLASS IMPLEMENTATIONS
  //------------------------------------------------------------------------------------
  // SceneDetection implementation
  SceneDetection::SceneDetection(const int &_imgHeight, const int &_imgWidth)
  {
    imageHeight = _imgHeight;
    imageWidth = _imgWidth;
  }

  std::vector<float> SceneDetection::getCannyVec() const
  {
    return cannyVectorBuffer;
  }

  std::vector<float> SceneDetection::getColorVec() const
  {
    return colorVectorBuffer;
  }

  std::vector<int> SceneDetection::predict(const std::vector<cv::Mat> &vidArray,
                                           const float &cannyThreshold,
                                           const int & minSceneLen)
  {
    // check input size
    if (vidArray[0].cols != imageWidth || vidArray[0].rows != imageHeight) {
      string errStr =  "[ERROR] video array does not match object height and size\n";
      throw std::length_error(errStr);
    }

    // Canny detections
    std::vector<int> cannyDets = sceneDetEdges(vidArray, cannyThreshold, minSceneLen);

    // Color detections

    return cannyDets;
  }

  std::vector<int> SceneDetection::predict(const std::string &vidPath,
                                           const float &cannyThreshold,
                                           const int &minSceneLen)
  {
    // TODO ugly but more efficient to do all types of detections in the same loop
    // clear the vector buffer
    cannyVectorBuffer.clear();

    // Default the motion iter
    // TODO maybe make this tunable
    const int motionIter = 6;
    const int diffRadius = 6;

    // Open video for reading
    cout << "[INFO] Opening video for reading" << endl;
    cv::VideoCapture cap(vidPath);
    if (!cap.isOpened()) {
      throw std::runtime_error("[ERROR] unable to open video");
    }

    // Initialize containers and downsampling stuff
    std::vector<int> detectedScene = {0};
    FrameBuffer frameBuffer(2);
    Size sampleSize(imageWidth, imageHeight);

    // Iterate through frames
    int i = 0;
    for (;;)
      {
        Mat frame;
        cap >> frame;
        if (frame.empty()) { break; }

        // Downsample image
        cv::resize(frame, frame, sampleSize);

        frameBuffer.add(frame);

        if (frameBuffer.full()) {
          double p = _frameDiffEdge(frameBuffer.frames[0],
                                    frameBuffer.frames[1],
                                    motionIter,
                                    diffRadius);

          cannyVectorBuffer.push_back(p);

          bool lessThanThresh = p > cannyThreshold;
          bool sceneLenOK = (i - detectedScene[detectedScene.size()-1]) > minSceneLen;
          bool isTransition = lessThanThresh && sceneLenOK;

          if (isTransition) { detectedScene.push_back(i); }
        }
        i++;
      }
    cap.release();

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
  Mat testImg(10, 10, CV_8UC1, Scalar(255));
  cv::rectangle(testImg, Point(2, 2), Point(7, 7), Scalar(0, 0, 0), -1);
  // cout << "testImg = " << endl << " " << testImg << endl << endl;

  Mat testCanny;
  transdet::_customCanny(testImg, testCanny);
  // cout << "testCanny = " << endl << " " << testCanny << endl << endl;

  // cout << "testing roll" << endl;
  // Mat testRoll = testImg.clone();
  // Mat tmp = transdet::rollCvMat(testRoll, 11, 0);
  // Mat rollRes = transdet::rollCvMat(tmp, 11, 1);

  Mat testImg2 = testImg.clone();
  // no shift
  double d1 = transdet::_hammingDist(testImg, testImg2);

  // vert shift
  Mat pSh = testImg2.clone();
  pSh = transdet::rollCvMat(pSh, 1, 0);
  pSh = transdet::rollCvMat(pSh, 1, 1);
  double d2 = transdet::_hammingDist(testImg, pSh);

  // pos shift
  Mat nSh = testImg2.clone();
  nSh = transdet::rollCvMat(nSh, -2, 0);
  nSh = transdet::rollCvMat(nSh, -2, 1);
  double d3 = transdet::_hammingDist(testImg, nSh);

  if (d1 == 0.0 && d2 == 0.22 && d3 == 0.4) {
    cout << "[SUCCESS] _hammingDist" << endl;
  } else {
    cout << "[FAILED] _hammingDist" << endl;
    cout << d1 << " " << d2 << " " << d3 << endl;
  }

  //------------------------------------------------------------------------------------

  Mat motTest = transdet::rollCvMat(testImg2, -2, 0);
  motTest = transdet::rollCvMat(motTest, -2, 1);
  Point motion = transdet::globalEdgeMotion(testImg, motTest, 6);

  if (motion.x == 2 && motion.y == 2) {
    cout << "[SUCCESS] globalEdgeMotion" << endl;
  } else {
    cout << "[FAILED] globalEdgeMotion" << endl;
    cout << "motion: [" << motion.x << ", " << motion.y << "]\n";
  }

  //------------------------------------------------------------------------------------

  transdet::FrameBuffer fBuff(2);
  fBuff.add(testImg);
  fBuff.add(motTest);

  bool fll = fBuff.full();
  int len = fBuff.length();

  if (fll && len == 2) {
    cout << "[SUCCESS] FrameBuffer" << endl;
  } else {
    cout << "[FAILED] FrameBuffer" << endl;
    cout << "len: " << len << endl;
  }

  Mat mt1, mt2, mt1r, mt2r;
  mt1 = cv::imread("../testims/mt1.jpg");
  mt2 = cv::imread("../testims/mt2.jpg");
  Size dnsz(100, 100);
  cv::resize(mt1, mt1r, dnsz);
  cv::resize(mt2, mt2r, dnsz);

  double d = transdet::_frameDiffEdge(mt1r, mt2r, 6, 6);
  cout << "d: " << d <<endl;

  //------------------------------------------------------------------------------------

  // Test video
  string vidPath = "/Users/mike/Desktop/test.mp4";

  // Initialize scene detector
  transdet::SceneDetection detector(100, 100);

  auto transitions = detector.predict(vidPath, 0.9, 2);

  cout << "<";
  for (auto && i : transitions)
    {
      cout << " " << i << " ";
    }
  cout << ">" << endl;

  cout << "<";
  for (auto && i : detector.getCannyVec())
    {
      cout << " " << i << " ";
    }
  cout << ">" << endl;

  //------------------------------------------------------------------------------------

  return 0;
}

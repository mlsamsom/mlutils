#ifndef VASE_tRans_
#define VASE_tRans_

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

namespace transdet
{
  // HELPERS
  //------------------------------------------------------------------------------------

  /*
   * Compute the median value of a matrix
   *
   * @param input an input image
   * @param nVals number of bins for the histogram
   * @returns median of a matrix
   */
  double _medianMat(cv::Mat &input, int &nVals);

  //------------------------------------------------------------------------------------

  /*
   * Calculates the percent difference between 2 images
   * Performs a binary dilation on each image and returns
   * (im1 & im2) / sum(im1)
   *
   * @param &canny1 A canny transformed image
   * @param &canny2 A canny transformed image
   * @param &num_iter number of times to perform dilation operation
   * @returns the percent difference between canny1 and canny2
   */
   double _frameDiff(const cv::Mat &canny1, const cv::Mat &canny2, const int &num_iter);

  //------------------------------------------------------------------------------------

  /*
   * Custom canny threshold detector
   * uses a median method to calculate the low and high threshold
   * using a pass by reference to conform to the OpenCV style
   *
   * @param &src grayscale image
   * @param &dst empty Mat
   */
  void _customCanny(const cv::Mat &src, cv::Mat &dst);

  //------------------------------------------------------------------------------------

  /*
   * Compute hamming distance between 2 binary images for a given x and y shift
   *
   * @param &binImg1 a binary image
   * @param &binImg2 a binary image
   * @param &xShift the radius shift in the x direction
   * @param &yShift the radius shift in the y direction
   * @returns the hamming distance
   */
  double _hammingDist(cv::Mat &binImg1, cv::Mat &binImg2, int &xShift, int &yShift);

  //------------------------------------------------------------------------------------

  /*
   * Compute the Hausdorff image between 2 binary images
   *
   * @param &binImg1 a binary image
   * @param &binImg2 a binary image
   * @param &xShift the radius shift in the x direction
   * @param &yShift the radius shift in the y direction
   * @returns the hamming distance
   */
  double _hausdorffDist(cv::Mat &binImg1, cv::Mat &binImg2, int &xShift, int &yShift);

  //------------------------------------------------------------------------------------

  // MAIN FUNCTIONS
  //------------------------------------------------------------------------------------

  /*
   * Global motion estimation using edge features
   * inputs must be canny edge images
   *
   * @param &canny1
   * @param &canny2
   * @param &radius search radius for measuring correspondence
   * @returns a vector of the motion required to minimize the edge
              distances between canny1 and canny2
   */
  std::vector<cv::Point> globalEdgeMotion(const cv::Mat &canny1,
                                          const cv::Mat &canny2,
                                          const int &radius)

  //------------------------------------------------------------------------------------

  /*
   * Detects scene transitions using edge deviation of a Canny transformed image
   * algorithm taken from Mai et al 1995
   *
   * @param &vidArray a video array
   * @param &threshold percent difference threshold
   * @param &min_scene_len estimate on the minimum scene length
   * @returns a list of frames that are the beginning of given scenes
   */
  std::vector<int> sceneDetEdges(const std::vector<cv::Mat> &vidArray,
                                 const float &threshold,
                                 const int &minSceneLen);

  //------------------------------------------------------------------------------------

  /*
   * Detects scene transitions using difference in color histograms
   *
   * @param &vidArray a video array
   * @param &threshold percent difference threshold
   * @param &min_scene_len estimate on the minimum scene length
   * @returns a list of frames that are the beginning of given scenes
   */
  std::vector<int> sceneDetColors(const std::vector<cv::Mat> &vidArray,
                                  const float &threshold,
                                  const int &minSceneLen);

  //------------------------------------------------------------------------------------

} // transdet

#endif  // VASE_tRans_

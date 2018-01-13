#ifndef VASE_tRans_
#define VASE_tRans_

#include <iostream>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>

namespace transdet
{
  // CLASSES

  //------------------------------------------------------------------------------------

  /*
   * Container for frames for difference streaming
   */
   class FrameBuffer
   {
   private:
     int bufferSize;
     int currentLen = 0;

   public:
     std::vector<cv::Mat> frames;

     // Constructor
     FrameBuffer(int _buffSz)
     {
       bufferSize = _buffSz;
     }

     // add a frame to buffer
     void add(cv::Mat inFrame)
     {
       if (currentLen < bufferSize) {
         frames.push_back(inFrame);
       }else if (currentLen > bufferSize) {
         throw std::out_of_range("FrameBuffer overflowed somehow");
       } else {
         remove();
         frames.push_back(inFrame);
       }
       currentLen++;
     }

     // remove first element added
     void remove()
     {
       frames.erase(frames.begin());
       currentLen--;
     }

     int length()
     {
       return currentLen;
     }

     bool full()
     {
       return currentLen == bufferSize;
     }
   };

  //------------------------------------------------------------------------------------

  // FUNCTION SIGNATURES
  // HELPERS
  //------------------------------------------------------------------------------------

  /*
   * Roll an OpenCV matrix over by axis
   *
   * TODO figure out how to re-write the header instead of making copies
   * @param &a an opencv image matrix
   * @param &shift the amount of pixels to shift by
   * @param &axis the axis in which to perform shift (0 (vert) or 1 (horiz))
   */
  cv::Mat rollCvMat(const cv::Mat &a,
                    const int &shift,
                    const int &axis);

  //------------------------------------------------------------------------------------

  /*
   * Compute the median value of a channel matrix
   * helper function for median mat
   *
   * @param input an input image
   * @returns median of a matrix
   */
  double _medianChannel(const cv::Mat &input);

  //------------------------------------------------------------------------------------
  /*
   * Compute the median value of a matrix
   * only works for 1 or 3 channel images
   *
   * @param input an input image
   * @param nVals number of bins for the histogram
   * @returns median of a matrix
   */
  double _medianMat(const cv::Mat &input);

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
  double _hammingDist(const cv::Mat &binImg1,
                      const cv::Mat &binImg2);

  //------------------------------------------------------------------------------------

  /*
   * Compute the Hausdorff image between 2 binary images
   *
   * @param &binImg1 a binary image const
   * @param &binImg2 a binary image const
   * @param &xShift the radius shift in the x direction
   * @param &yShift the radius shift in the y direction
   * @returns the hamming distance
   */
  double _hausdorffDist(cv::Mat &binImg1, cv::Mat &binImg2, int &xShift, int &yShift);

  //------------------------------------------------------------------------------------

  /*
   * Use canny edge detector to determine a metric of difference between 2 frames
   *
   * @param &currentImage const
   * @param &nextImage const
   * @param &threshold const
   * @param &minSceneLen const
   * @returns a distance metric between frames
   */
  double _frameDiffEdge(const cv::Mat &currentImage,
                        const cv::Mat &nextImage,
                        const int &motionIter,
                        const int &diffRadius);

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
  cv::Point globalEdgeMotion(const cv::Mat &canny1,
                             const cv::Mat &canny2,
                             const int &radius);

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
  * Detects scene transitions using edge deviation of a Canny transformed image
  * algorithm taken from Mai et al 1995 overridden from above to stream frames
  * without keeping everything in memory
  *
  * @param &vidPath path to a video file
  * @param &threshold percent difference threshold
  * @param &min_scene_len estimate on the minimum scene length
  * @returns a list of frames that are the beginning of given scenes
  */
  std::vector<int> sceneDetEdges(const std::string &vidPath,
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

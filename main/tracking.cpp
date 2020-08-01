#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d.hpp"
#include "stdio.h"

using namespace cv::xfeatures2d;

using namespace std;
using namespace cv;

int main(){

  // Create a VideoCapture object and open the input file
  // If the input is the web camera, pass 0 instead of the video file name
  VideoCapture cap("/home/ecn/tracking/build/video1.mp4");

  // Check if camera opened successfully
  if(!cap.isOpened()){
    cout << "Error opening video stream or file" << endl;
    return -1;
  }
  Mat frame_roi;
  int zhen = 1;
  //std::vector<KeyPoint> keypoints_object, keypoints_scene;
  //Mat descriptors_object, descriptors_scene;
  while(1){

    Mat frame;
    // Capture frame-by-frame
    cap >> frame;

    // If the frame is empty, break immediately
    if (frame.empty())
      break;

    // Display the resulting frame
    //imshow( "Frame", frame );

    // step 1: changing the image to grayscale
    cvtColor(frame, frame, CV_BGR2GRAY);
    imshow( "Frame", frame );
    //step 2: extracting features at the first frame
    //Mat frame_roi;

    if(zhen == 1)
    {
        //region of interest
        Rect rect(24,44,172,166);
        frame_roi = frame(rect);
        //imshow("ROI",frame_roi);
        zhen++;
    }

        // feature keypoints
        int minHessian = 800;
        Ptr<SURF> detector = SURF::create( minHessian );
        std::vector<KeyPoint> keypoints_object, keypoints_scene;
        Mat descriptors_object, descriptors_scene;
        detector->detectAndCompute( frame_roi, noArray(), keypoints_object, descriptors_object );
        detector->detectAndCompute( frame, noArray(), keypoints_scene, descriptors_scene );

        //imshow("keypoints",keypoints_object);
        //-- Step 3: Matching descriptor vectors with a FLANN based matcher
        // Since SURF is a floating-point descriptor NORM_L2 is used
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
        std::vector< std::vector<DMatch> > knn_matches;
        matcher->knnMatch( descriptors_object, descriptors_scene, knn_matches, 2 );
        //-- Filter matches using the Lowe's ratio test
        const float ratio_thresh = 0.75f;
        std::vector<DMatch> good_matches;
        // calculating the distance between descriptors
        for (size_t i = 0; i < knn_matches.size(); i++)
        {
            if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
            {
                good_matches.push_back(knn_matches[i][0]);
            }
        }
        //-- Draw matches
        Mat img_matches;
        drawMatches( frame_roi, keypoints_object, frame, keypoints_scene, good_matches, img_matches, Scalar::all(-1),
                     Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        //-- Localize the object
        std::vector<Point2f> obj;
        std::vector<Point2f> scene;
        for( size_t i = 0; i < good_matches.size(); i++ )
        {
            //-- Get the keypoints from the good matches
            obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
            scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
        }
        //step 4: from feature map computing homography
        Mat H = findHomography( obj, scene, RANSAC );
        //-- Get the corners from the image_1 ( the object to be "detected" )
        std::vector<Point2f> obj_corners(4);
        obj_corners[0] = Point2f(0, 0);
        obj_corners[1] = Point2f( (float)frame_roi.cols, 0 );
        obj_corners[2] = Point2f( (float)frame_roi.cols, (float)frame_roi.rows );
        obj_corners[3] = Point2f( 0, (float)frame_roi.rows );
        std::vector<Point2f> scene_corners(4);
        perspectiveTransform( obj_corners, scene_corners, H);
        //-- Draw lines between the corners (the mapped object in the scene - image_2 )
        line( img_matches, scene_corners[0] + Point2f((float)frame_roi.cols, 0),
              scene_corners[1] + Point2f((float)frame_roi.cols, 0), Scalar(0, 255, 0), 4 );
        line( img_matches, scene_corners[1] + Point2f((float)frame_roi.cols, 0),
              scene_corners[2] + Point2f((float)frame_roi.cols, 0), Scalar( 0, 255, 0), 4 );
        line( img_matches, scene_corners[2] + Point2f((float)frame_roi.cols, 0),
              scene_corners[3] + Point2f((float)frame_roi.cols, 0), Scalar( 0, 255, 0), 4 );
        line( img_matches, scene_corners[3] + Point2f((float)frame_roi.cols, 0),
              scene_corners[0] + Point2f((float)frame_roi.cols, 0), Scalar( 0, 255, 0), 4 );
        //-- Show detected matches
        imshow("Good Matches & Object detection", img_matches );
        //waitKey();

    //step 3:
        imshow( "Frame", frame );

    // Press  ESC on keyboard to exit
    char c=(char)waitKey(25);
    if(c==27)
      break;

  }

  // When everything done, release the video capture object
  cap.release();

  // Closes all the frames
  destroyAllWindows();

  return 0;
}

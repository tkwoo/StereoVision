#include <opencv2\opencv.hpp>
#include <opencv2\ximgproc.hpp>
#include <opencv2\imgcodecs.hpp>
#include <opencv2\viz.hpp>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::ximgproc;

int main()
{
	viz::Viz3d plot3d("Coordinate Frame");
	plot3d.showWidget("Coordinate Widget", viz::WCoordinateSystem());

	Mat left_for_matcher, right_for_matcher;
	Mat imgLeftDisp, imgRightDisp, disp8;
	Mat filtered_disp;
	Rect ROI;
	Ptr<DisparityWLSFilter> wls_filter;
	double matching_time, filtering_time;

	int nSADWindowSize = 9;
	int nNumberOfDisparities = 80;
	double lambda = 8000.;
	double sigma = 1.5;
	double vis_mult = 3.0;

	Ptr<StereoBM> left_bm = StereoBM::create(0, 0);

	left_bm->create(nNumberOfDisparities, nSADWindowSize);
	left_bm->setPreFilterCap(31);
	left_bm->setBlockSize(nSADWindowSize > 0 ? nSADWindowSize : 9);
	left_bm->setMinDisparity(0);
	left_bm->setNumDisparities(nNumberOfDisparities);
	left_bm->setTextureThreshold(10);
	left_bm->setUniquenessRatio(15);
	left_bm->setSpeckleWindowSize(100);
	left_bm->setSpeckleRange(32);
	left_bm->setDisp12MaxDiff(1);


	bool flgFilter = true; 

	while (!plot3d.wasStopped()){
		Mat imgLeft = imread("Left_923730u.pgm", 0);
		Mat imgRight = imread("Right_923730u.pgm", 0);
		Mat conf_map = Mat(imgLeft.rows, imgLeft.cols, CV_8U);
		conf_map = Scalar(255);

		if (flgFilter = true)
		{
			left_for_matcher = imgLeft.clone();
			right_for_matcher = imgRight.clone();

			wls_filter = createDisparityWLSFilter(left_bm);
			Ptr<StereoMatcher> right_bm = createRightMatcher(left_bm);

			matching_time = (double)getTickCount();
			left_bm->compute(left_for_matcher, right_for_matcher, imgLeftDisp);
			right_bm->compute(right_for_matcher, left_for_matcher, imgRightDisp);
			matching_time = ((double)getTickCount() - matching_time) / getTickFrequency();

			wls_filter->setLambda(lambda);
			wls_filter->setSigmaColor(sigma);

			filtering_time = (double)getTickCount();
			wls_filter->filter(imgLeftDisp, imgLeft, filtered_disp, imgRightDisp);
			filtering_time = ((double)getTickCount() - filtering_time) / getTickFrequency();
			conf_map = wls_filter->getConfidenceMap();

			ROI = wls_filter->getROI();

			imshow("left", imgLeft);
			getDisparityVis(imgLeftDisp, disp8, vis_mult);
			imshow("raw disp", disp8);
			Mat imgFilteredDisp8;
			filtered_disp.convertTo(imgFilteredDisp8, CV_8U, 255 / (nNumberOfDisparities*16.));
			imshow("filtered disp", imgFilteredDisp8);

			Mat xyz;
			vector<Point3f> depthPts;
			vector<uchar> colorPts;
			for (int i = 0; i < imgFilteredDisp8.rows; i++){
				for (int j = 0; j < imgFilteredDisp8.cols; j++){
					if (imgFilteredDisp8.at<uchar>(i, j) == 0) continue;
					float z = 1200 * 0.25 / ((float)imgFilteredDisp8.at<uchar>(i, j)*(float)nNumberOfDisparities / 255);
					if (z < 60 && z > 0){
						Point3f temp_point((j - 320)*z / 1200, (i - 240)*z / 1200, z);
						depthPts.push_back(temp_point);
						colorPts.push_back(imgLeft.at<uchar>(i, j));
					}
				}
			}

			cv::viz::WCloud cloud_widget = cv::viz::WCloud(depthPts, colorPts);
			cloud_widget.setRenderingProperty(cv::viz::POINT_SIZE, 2);

			plot3d.showWidget("ref_cloud", cloud_widget);

		}

		if (waitKey(1) == 27) return 1;
		plot3d.spinOnce(1, true);
	}


	return 0;
}
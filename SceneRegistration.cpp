#include "SceneRegistration.h"
#include "Timer.h"
#include "Parameters.h"

std::vector<std::vector<float> > SceneRegistration::getDepth(RealsenseGrabber* grabber) {
	puts("!!!");
	const cv::Size BOARD_SIZE = cv::Size(9, 6);
	const int BOARD_NUM = BOARD_SIZE.width * BOARD_SIZE.height;
	const float GRID_SIZE = 0.028f;
	const int ITERATION = 10;
	const float INTERVAL = 0.0f;

	std::vector<cv::Point3f> objectPoints;
	for (int r = 0; r < BOARD_SIZE.height; r++) {
		for (int c = 0; c < BOARD_SIZE.width; c++) {
			objectPoints.push_back(cv::Point3f(c * GRID_SIZE, r * GRID_SIZE, 0));
		}
	}

	Transformation* depthTrans;
	Intrinsics* depthIntrinsics;
	Intrinsics* colorIntrinsics;
	UINT16** depthImages;
	RGBQUAD** colorImages;
	std::vector<cv::Point2f> sourcePoints;
	cv::Mat sourceColorMat(COLOR_H, COLOR_W, CV_8UC3);

	std::vector<std::vector<float> > depths;
	int cameras = grabber->getRGBD(depthImages, colorImages, depthTrans, depthIntrinsics, colorIntrinsics);
	for (int id = 0; id < cameras; id++) {
		std::vector<std::vector<cv::Point2f> > sourcePointsArray;
		for (int iter = 0; iter < ITERATION;) {
			
			RGBQUAD* source = colorImages[id];
			for (int i = 0; i < COLOR_H; i++) {
				for (int j = 0; j < COLOR_W; j++) {
					RGBQUAD color;
					color = source[i * COLOR_W + j];
					sourceColorMat.at<cv::Vec3b>(i, j) = cv::Vec3b(color.rgbRed, color.rgbGreen, color.rgbBlue);
				}
			}
			sourcePoints.clear();
			findChessboardCorners(sourceColorMat, BOARD_SIZE, sourcePoints, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);

			cv::Scalar color = cv::Scalar(0, 0, 255);
			if (sourcePoints.size() == BOARD_NUM) {
				color = cv::Scalar(0, 255, 255);
			}
			for (int i = 0; i < sourcePoints.size(); i++) {
				cv::circle(sourceColorMat, sourcePoints[i], 3, color, 2);
			}
			cv::imshow("Get Depth", sourceColorMat);

			char ch = cv::waitKey(1);
			if (int(ch) != -1) {
				iter = 0;
			}

			if (iter != -1 && sourcePoints.size() == BOARD_NUM) {
				iter++;
				sourcePointsArray.push_back(sourcePoints);
			}
		}

		std::vector<std::vector<cv::Point3f> > objectPointsArray;
		for (int i = 0; i < sourcePointsArray.size(); i++) {
			objectPointsArray.push_back(objectPoints);
		}

		cv::Mat cameraMatrix, distCoeffs;
		std::vector<cv::Mat> rvec, tvec;
		std::vector<float> reprojError;



		
		double rms = calibrateCamera(objectPointsArray,
			sourcePointsArray,
			sourceColorMat.size(),
			cameraMatrix,
			distCoeffs,
			rvec,
			tvec,
			CV_CALIB_FIX_K4 | CV_CALIB_FIX_K5);
		std::cout << "camera  " << cameraMatrix << std::endl;

		if (!(cv::checkRange(cameraMatrix) && cv::checkRange(distCoeffs))) {
			std::cout << "Calibration failed\n";
		}

		cv::Mat rv(3, 1, CV_64FC1);
		cv::Mat tv(3, 1, CV_64FC1);

		cv::Mat sourceCameraMatrix(cv::Size(3, 3), CV_32F);
		sourceCameraMatrix.at<float>(0, 0) = colorIntrinsics[0].fx;
		sourceCameraMatrix.at<float>(1, 1) = colorIntrinsics[0].fy;
		sourceCameraMatrix.at<float>(0, 2) = colorIntrinsics[0].ppx;
		sourceCameraMatrix.at<float>(1, 2) = colorIntrinsics[0].ppy;

		solvePnP(objectPointsArray[0], sourcePointsArray[0], sourceCameraMatrix, distCoeffs, rv, tv);
		for (int i = 0; i < sourcePointsArray[0].size(); i++)
			std::cout << sourcePointsArray[0][i] << " "; std::cout << std::endl;
		std::cout << rv << std::endl;
		std::cout << tv << std::endl;
		std::vector<float> depth;
		for (int i = 0; i < objectPoints.size(); i++)
			depth.push_back(cv::norm(objectPoints[i] - cv::Point3f(tv)));
		depths.push_back(depth);
	}
	return depths;
}


Transformation SceneRegistration::align(RealsenseGrabber* grabber, Transformation* colorTrans)
{
	const cv::Size BOARD_SIZE = cv::Size(9, 6);
	const int BOARD_NUM = BOARD_SIZE.width * BOARD_SIZE.height;
	const float GRID_SIZE = 0.028f;
	const int ITERATION = 20;
	const float INTERVAL = 0.0f;
	const int corners[4] = { 0, 8, 53, 45 };


	UINT16** depthImages;
	RGBQUAD** colorImages;
	Transformation* depthTrans;
	Intrinsics* depthIntrinsics;
	Intrinsics* colorIntrinsics;
	std::vector<cv::Point2f> sourcePoints;
	std::vector<cv::Point2f> targetPoints;
	cv::Mat sourceColorMat(COLOR_H, COLOR_W, CV_8UC3);
	cv::Mat targetColorMat(COLOR_H, COLOR_W, CV_8UC3);

	std::vector<cv::Point3f> objectPoints;
	for (int r = 0; r < BOARD_SIZE.height; r++) {
		for (int c = 0; c < BOARD_SIZE.width; c++) {
			objectPoints.push_back(cv::Point3f(c * GRID_SIZE, r * GRID_SIZE, 0));
		}
	}

	int cameras = -1;

	for (int targetId = 1; cameras == -1 || targetId < cameras; targetId++) {
		std::vector<std::vector<cv::Point2f> > sourcePointsArray;
		std::vector<std::vector<cv::Point2f> > targetPointsArray;
		std::vector<cv::Point2f> rects;
		Timer timer;
		for (int iter = 0; iter < ITERATION;) {
			cameras = grabber->getRGBD(depthImages, colorImages, depthTrans, depthIntrinsics, colorIntrinsics);
			RGBQUAD* source = colorImages[0];
			RGBQUAD* target = colorImages[targetId];
			for (int i = 0; i < COLOR_H; i++) {
				for (int j = 0; j < COLOR_W; j++) {
					RGBQUAD color;
					color = source[i * COLOR_W + j];
					sourceColorMat.at<cv::Vec3b>(i, j) = cv::Vec3b(color.rgbRed, color.rgbGreen, color.rgbBlue);
					color = target[i * COLOR_W + j];
					targetColorMat.at<cv::Vec3b>(i, j) = cv::Vec3b(color.rgbRed, color.rgbGreen, color.rgbBlue);
				}
			}

			sourcePoints.clear();
			targetPoints.clear();
			findChessboardCorners(sourceColorMat, BOARD_SIZE, sourcePoints, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);
			findChessboardCorners(targetColorMat, BOARD_SIZE, targetPoints, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);
			cv::Scalar color = cv::Scalar(0, 0, 255);
			if (sourcePoints.size() == BOARD_NUM) {
				if (timer.getTime() > INTERVAL) {
					color = cv::Scalar(0, 255, 0);
				}
				else {
					color = cv::Scalar(0, 255, 255);
				}
			}
			for (int i = 0; i < sourcePoints.size(); i++) {
				cv::circle(sourceColorMat, sourcePoints[i], 3, color, 2);
			}
			for (int i = 0; i < targetPoints.size(); i++) {
				cv::circle(targetColorMat, targetPoints[i], 3, color, 2);
			}
			cv::Mat mergeImage;
			bool valid = (iter != -1 && sourcePoints.size() == BOARD_NUM && targetPoints.size() == BOARD_NUM);
			cv::Point2f center;
			for (int i = 0; i < rects.size(); i += 5) {
				for (int j = 0; j < 4; j++) {
					cv::line(sourceColorMat, rects[i + j], rects[i + (j + 1) % 4], cv::Scalar(0, 255, 0), 2);
				}
			}
			if (valid) {
				center = (sourcePoints[corners[0]] +
					sourcePoints[corners[1]] +
					sourcePoints[corners[2]] +
					sourcePoints[corners[3]]) / 4;
				for (int i = 4; i < rects.size(); i += 5) {
					if (cv::norm(rects[i] - center) < 20)
						valid = false;
				}
				cv::Scalar color = valid ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
				for (int i = 0; i < 4; i++) {
					cv::line(sourceColorMat, sourcePoints[corners[i]], sourcePoints[corners[(i + 1) % 4]], color, 2);
				}
				
			}
			cv::hconcat(sourceColorMat, targetColorMat, mergeImage);
			cv::imshow("Calibration", mergeImage);

			char ch = cv::waitKey(1);
			if (int(ch) != -1) {
				iter = 0;
			}
			std::cout << iter << std::endl;
			if (valid) {
				iter++;
				sourcePointsArray.push_back(sourcePoints);
				targetPointsArray.push_back(targetPoints);
				for (int i = 0; i < 4; i++) {
					rects.push_back(sourcePoints[corners[i]]);
				}
				rects.push_back(center);
				timer.reset();
			}
		}

		std::vector<std::vector<cv::Point3f> > objectPointsArray;
		for (int i = 0; i < sourcePointsArray.size(); i++) {
			objectPointsArray.push_back(objectPoints);
		}

		cv::Mat sourceCameraMatrix(cv::Size(3, 3), CV_32F);
		sourceCameraMatrix.at<float>(0, 0) = colorIntrinsics[0].fx;
		sourceCameraMatrix.at<float>(1, 1) = colorIntrinsics[0].fy;
		sourceCameraMatrix.at<float>(0, 2) = colorIntrinsics[0].ppx;
		sourceCameraMatrix.at<float>(1, 2) = colorIntrinsics[0].ppy;
		sourceCameraMatrix.at<float>(2, 2) = 1;
		cv::Mat targetCameraMatrix(cv::Size(3, 3), CV_32F);
		targetCameraMatrix.at<float>(0, 0) = colorIntrinsics[targetId].fx;
		targetCameraMatrix.at<float>(1, 1) = colorIntrinsics[targetId].fy;
		targetCameraMatrix.at<float>(0, 2) = colorIntrinsics[targetId].ppx;
		targetCameraMatrix.at<float>(1, 2) = colorIntrinsics[targetId].ppy;
		targetCameraMatrix.at<float>(2, 2) = 1;

		cv::Mat sourceDistCoeffs;
		cv::Mat targetDistCoeffs;
		cv::Mat rotation, translation, essential, fundamental;
		double rms = cv::stereoCalibrate(
			objectPointsArray,
			sourcePointsArray,
			targetPointsArray,
			sourceCameraMatrix,
			sourceDistCoeffs,
			targetCameraMatrix,
			targetDistCoeffs,
			cv::Size(COLOR_H, COLOR_W),
			rotation,
			translation,
			essential,
			fundamental
		);

		colorTrans[targetId] = Transformation((double*)rotation.data, (double*)translation.data);
	}
	cv::destroyAllWindows();
}

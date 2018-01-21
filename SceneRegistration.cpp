#include "SceneRegistration.h"
#include "Timer.h"
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>

SceneRegistration::SceneRegistration() {

}

SceneRegistration::~SceneRegistration() {

}

void SceneRegistration::transform(const Eigen::Matrix4f & mat, const pcl::PointXYZ & p, pcl::PointXYZ & out) {
	out.x = mat(0, 0)*p.x + mat(0, 1)*p.y + mat(0, 2)*p.z + mat(0, 3);
	out.y = mat(1, 0)*p.x + mat(1, 1)*p.y + mat(1, 2)*p.z + mat(1, 3);
	out.z = mat(2, 0)*p.x + mat(2, 1)*p.y + mat(2, 2)*p.z + mat(2, 3);
}

Eigen::Matrix4f SceneRegistration::align(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr source, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr target)
{
	Timer timer;
	timer.reset();

	Eigen::Matrix4f transformation;
	transformation.setIdentity();
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr sourcePoints(new pcl::PointCloud<pcl::PointXYZRGB>());
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr targetPoints(new pcl::PointCloud<pcl::PointXYZRGB>());
	pcl::copyPointCloud(*source, *sourcePoints);
	pcl::copyPointCloud(*target, *targetPoints);
	pcl::PointCloud<pcl::Normal>::Ptr sourceNormals(new pcl::PointCloud<pcl::Normal>());
	pcl::PointCloud<pcl::Normal>::Ptr targetNormals(new pcl::PointCloud<pcl::Normal>());
	pcl::copyPointCloud(*source, *sourceNormals);
	pcl::copyPointCloud(*target, *targetNormals);

	std::cout << "Extracting SIFT Keypoints" << std::endl;
	pcl::PointCloud<pcl::PointWithScale>::Ptr sourceKeypointsScale(new pcl::PointCloud<pcl::PointWithScale>());
	pcl::PointCloud<pcl::PointWithScale>::Ptr targetKeypointsScale(new pcl::PointCloud<pcl::PointWithScale>());
	pcl::SIFTKeypoint<pcl::PointXYZRGB, pcl::PointWithScale> siftDetect;
	siftDetect.setScales(0.0025, 5, 5);
	siftDetect.setMinimumContrast(0.8);
	siftDetect.setInputCloud(sourcePoints);
	siftDetect.compute(*sourceKeypointsScale);
	siftDetect.setInputCloud(targetPoints);
	siftDetect.compute(*targetKeypointsScale);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr sourceKeypoints(new pcl::PointCloud<pcl::PointXYZRGB>());
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr targetKeypoints(new pcl::PointCloud<pcl::PointXYZRGB>());
	pcl::copyPointCloud(*sourceKeypointsScale, *sourceKeypoints);
	pcl::copyPointCloud(*targetKeypointsScale, *targetKeypoints);

	std::cout << "Calculating SHOT Descriptor" << std::endl;
	pcl::PointCloud<pcl::SHOT1344>::Ptr sourceDescr(new pcl::PointCloud<pcl::SHOT1344>());
	pcl::PointCloud<pcl::SHOT1344>::Ptr targetDescr(new pcl::PointCloud<pcl::SHOT1344>());
	pcl::SHOTColorEstimationOMP<pcl::PointXYZRGB, pcl::Normal, pcl::SHOT1344> descrEst;
	descrEst.setRadiusSearch(0.05);
	descrEst.setInputCloud(sourceKeypoints);
	descrEst.setSearchSurface(sourcePoints);
	descrEst.setInputNormals(sourceNormals);
	descrEst.compute(*sourceDescr);
	descrEst.setInputCloud(targetKeypoints);
	descrEst.setSearchSurface(targetPoints);
	descrEst.setInputNormals(targetNormals);
	descrEst.compute(*targetDescr);

	std::cout << "Searching Correspondences";
	pcl::CorrespondencesPtr corrs(new pcl::Correspondences());
	pcl::search::KdTree<pcl::SHOT1344> kdTree;
	kdTree.setInputCloud(targetDescr);
#pragma omp parallel for
	for (int i = 0; i < sourceDescr->size(); i++)
	{
		if (pcl_isnan(sourceDescr->at(i).descriptor[0])) {
			continue;
		}
		std::vector<int> targetIndex(2);
		std::vector<float> sqrDist(2);
		int found = kdTree.nearestKSearch(sourceDescr->points[i], 2, targetIndex, sqrDist);

		if (found == 2 && sqrDist[0] / sqrDist[1] < 0.64 && sqrDist[0] < 0.25) {
			corrs->push_back(pcl::Correspondence(i, targetIndex[0], sqrDist[0]));
		}
	}
	std::cout << " = " << corrs->size() << std::endl;

	std::cout << "SANSAC rejecting";
	pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZRGB> rejector;
	rejector.setInputSource(sourceKeypoints);
	rejector.setInputTarget(targetKeypoints);
	rejector.setInlierThreshold(0.01);
	rejector.setInputCorrespondences(corrs);
	rejector.getCorrespondences(*corrs);
	std::cout << ", Remain = " << corrs->size() << std::endl;

	std::cout << "Estimating Transformation" << std::endl;
	estimateRigidTransformation(*sourceKeypoints, *targetKeypoints, *corrs, transformation);

	// Analaysis
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformedKeypoints(new pcl::PointCloud<pcl::PointXYZRGB>());
	pcl::transformPointCloud(*sourceKeypoints, *transformedKeypoints, transformation);
	float sum = 0;
	for (int k = 0; k < corrs->size(); k++) {
		pcl::PointXYZRGB* pt1 = &transformedKeypoints->at(corrs->at(k).index_query);
		pcl::PointXYZRGB* pt2 = &targetKeypoints->at(corrs->at(k).index_match);
		sum  += sqrt((pt1->x - pt2->x) * (pt1->x - pt2->x) + (pt1->y - pt2->y) * (pt1->y - pt2->y) + (pt1->z - pt2->z) * (pt1->z - pt2->z));
	}
	std::cout << "Average Distance = " << sum / corrs->size() << std::endl;

	timer.outputTime();

	/*
	// Visualization

	pcl::transformPointCloud(*source, *source, transformation);
	pcl::transformPointCloud(*sourceKeypoints, *sourceKeypoints, transformation);

	for (int i = 0; i < sourceKeypoints->size(); i++) {
		sourceKeypoints->points[i].r = sourceKeypoints->points[i].g = sourceKeypoints->points[i].b = 0;
		sourceKeypoints->points[i].r = 255;
	}
	for (int i = 0; i < targetKeypoints->size(); i++) {
		targetKeypoints->points[i].r = targetKeypoints->points[i].g = targetKeypoints->points[i].b = 0;
		targetKeypoints->points[i].g = 255;
	}

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Point Cloud Viewer"));
	viewer->setCameraPosition(0.0, 0.0, -2.0, 0.0, 0.0, 0.0);

	while (!viewer->wasStopped()) {
		viewer->spinOnce();

		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());

		pcl::copyPointCloud(*sourceKeypoints, *cloud);
		if (!viewer->updatePointCloud(cloud, "1")) {
			viewer->addPointCloud(cloud, "1");
		}
		pcl::copyPointCloud(*targetKeypoints, *cloud);
		if (!viewer->updatePointCloud(cloud, "2")) {
			viewer->addPointCloud(cloud, "2");
		}
		pcl::copyPointCloud(*source, *cloud);
		if (!viewer->updatePointCloud(cloud, "3")) {
			viewer->addPointCloud(cloud, "3");
		}
		pcl::copyPointCloud(*target, *cloud);
		if (!viewer->updatePointCloud(cloud, "4")) {
			viewer->addPointCloud(cloud, "4");
		}

		for (int i = 0; i < corrs->size(); i++) {
			viewer->addLine<pcl::PointXYZRGB, pcl::PointXYZRGB>(sourceKeypoints->points[(*corrs)[i].index_query], targetKeypoints->points[(*corrs)[i].index_match], 200, 200, 0, "line" + i);
		}
	}*/

	return transformation;
}

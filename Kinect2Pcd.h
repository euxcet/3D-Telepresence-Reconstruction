#ifndef KINECT_2_PCD
#define KINECT_2_PCD

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/grabber.h>

class Kinect2Pcd {
private:
	typedef pcl::PointXYZRGB PointType;

	boost::mutex mutex;
	boost::shared_ptr<pcl::Grabber> grabber;
	boost::signals2::connection connection;

	//Result
	bool updated;
	pcl::PointCloud<PointType>::Ptr cloud;

public:
	Kinect2Pcd();
	~Kinect2Pcd();
	bool isUpdated();
	pcl::PointCloud<PointType>::Ptr getPointCloud();
};

#endif
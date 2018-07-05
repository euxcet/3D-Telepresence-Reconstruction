#include "Timer.h"
#include "SceneRegistration.h"
#include "TsdfVolume.h"
#include "Transmission.h"
#include "RealsenseGrabber.h"
#include "Parameters.h"
#include <pcl/visualization/cloud_viewer.h>
#include <windows.h>

#define CREATE_EXE
//#define TRANSMISSION

byte* buffer = NULL;
RealsenseGrabber* grabber = NULL;
TsdfVolume* volume = NULL;
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;

UINT16** depthImages;
RGBQUAD** colorImages;
Transformation* colorTrans = NULL;

#ifdef TRANSMISSION
Transmission* transmission = NULL;
#endif

void registration() {
	SceneRegistration::align(grabber, colorTrans);
}

void keyboardEventOccurred(const pcl::visualization::KeyboardEvent& event) {
	if (event.getKeySym() == "r" && event.keyDown()) {
		registration();
	}
	else if (event.getKeySym() == "d" && event.keyDown()) {
		std::vector<std::vector<float> > depths = SceneRegistration::getDepth(grabber);
		std::cout << depths.size() << std::endl;
	}
}

void startViewer() {
	viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer("Point Cloud Viewer"));
	viewer->setCameraPosition(0.0, 0.0, -2.0, 0.0, 0.0, 0.0);
	viewer->registerKeyboardCallback(keyboardEventOccurred);
}

#ifdef TRANSMISSION
DWORD WINAPI TransmissionRecvThread(LPVOID pM)
{
#pragma omp parallel sections
{
	#pragma omp section
	{
		while (true) {
			Sleep(1);
			transmission->recvRGBD(depthList[1], colorList[1]);
		}
	}
}
	return 0;
}
#endif

void start() {
	omp_set_num_threads(4);
	omp_set_nested(6);
	int cudaDevices = 0;
	cudaGetDeviceCount(&cudaDevices);
	if (cudaDevices >= 2) {
		cudaSetDevice(1);
	}
	
	grabber = new RealsenseGrabber();
	cloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>());
	volume = new TsdfVolume(2, 2, 2, 0, 0, 1);
	buffer = new byte[MAX_VERTEX * sizeof(Vertex)];
	colorTrans = new Transformation[MAX_CAMERAS];

#ifdef TRANSMISSION
	transmission = new Transmission(true);
	CreateThread(NULL, 0, TransmissionRecvThread, NULL, 0, NULL);
#endif
}

void update() {
	Transformation* depthTrans;
	Intrinsics* depthIntrinsics;
	Intrinsics* colorIntrinsics;
	int cameras = grabber->getRGBD(depthImages, colorImages, depthTrans, depthIntrinsics, colorIntrinsics);

#ifdef TRANSMISSION
	// TODO
#else
	volume->integrate(buffer, cameras, depthImages, colorImages, depthTrans, colorTrans, depthIntrinsics, colorIntrinsics);
#endif
}

void stop() {
	if (grabber != NULL) {
		delete grabber;
	}
	if (volume != NULL) {
		delete volume;
	}
	if (buffer != NULL) {
		delete[] buffer;
	}
	if (colorTrans != NULL) {
		delete[] colorTrans;
	}
#ifdef TRANSMISSION
	if (transmission != NULL) {
		delete transmission;
	}
#endif
}

#ifdef CREATE_EXE

int main(int argc, char *argv[]) {
	start();
	while (1) {
		std::vector<std::vector<float> > depths = SceneRegistration::getDepth(grabber);
		std::cout << depths.size() << std::endl;
		float sum=  0;
		for (int j = 0; j < depths[0].size(); j++) {
			std::cout << depths[0][j] << " ";
			sum += depths[0][j];
		}
		std::cout << sum / depths[0].size() << std::endl;
		std::cout << std::endl;
	}
	system("pause");
	startViewer();
	while (!viewer->wasStopped()) {
		viewer->spinOnce();

		Timer timer;
		update();
		timer.outputTime();

		cloud = volume->getPointCloudFromMesh(buffer);
		if (!viewer->updatePointCloud(cloud, "cloud")) {
			viewer->addPointCloud(cloud, "cloud");
		}
	}
	stop();

	return 0;
}

#else
extern "C" {
	__declspec(dllexport) void callStart() {
		start();
	}

	__declspec(dllexport) byte* callUpdate() {
		update();
		return buffer;
	}

	__declspec(dllexport) void callRegistration() {
		registration();
	}

	__declspec(dllexport) void callStop() {
		stop();
	}
}
#endif

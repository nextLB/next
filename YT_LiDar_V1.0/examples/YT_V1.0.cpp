
#include "unitree_lidar_sdk.h"
#include <opencv2/opencv.hpp>
#include <cmath>
#include <ctime>

using namespace cv;
using namespace std;
using namespace unitree_lidar_sdk;

#define cloud_scan_num 18




int main(){
    // Initialize Lidar Object
    LidarReader* YTReader = createUnitreeLidarReader();
    string portName = "/dev/ttyUSB0";

    if (YTReader->initialize(cloud_scan_num, portName)){
        cout << "Lidar initialization failed! Exit here!" << endl;
        exit(-1);
    }
    else
        cout << "Lidar initialization succeed!" << endl;

    // Set Lidar Working Mode
    cout << "Set Lidar working mode to: NORMAL ... " << endl;
    YTReader->setLidarWorkingMode(NORMAL);
    sleep(1);

    while(true)
        if(YTReader->runParse() == VERSION)
        {
            cout << YTReader->getVersionOfFirmware().c_str();
            break;
        }


    // Parse PointCloud and IMU data
    MessageType result;
    std::string version;
    // set start time
    clock_t startTime = clock();
    double endTime = static_cast<double>(clock() - startTime) / CLOCKS_PER_SEC;

    while(endTime < 100)
    {
        result = YTReader->runParse();  // You need to call this function at least 1500Hz


        if (result == IMU)
            continue;
        else if(result == POINTCLOUD)
            for(size_t i = 0; i< YTReader->getCloud().points.size(); i++)
                cout << YTReader->getCloud().stamp << "  " << YTReader->getCloud().id << "  " << YTReader->getCloud().points.size() << "  " << YTReader->getCloud().ringNum << "  " << YTReader->getCloud().points[i].x << "  " << YTReader->getCloud().points[i].y << "  " << YTReader->getCloud().points[i].z << "  " << YTReader->getCloud().points[i].intensity << "  " << YTReader->getCloud().points[i].time << "  " << YTReader->getCloud().points[i].ring <<endl;


        endTime = static_cast<double>(clock() - startTime) / CLOCKS_PER_SEC;
//        cout << "目前激光雷达已经运行约: " << endTime << "秒" << endl;
    }


//    // Set Lidar Working Mode
//    cout << "Set Lidar working mode to: STANDBY ... " << endl;
//    YTReader->setLidarWorkingMode(STANDBY);
//    sleep(1);


    return 0;
}










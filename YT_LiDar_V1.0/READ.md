

# Unitree-based LiDAR should be included in the instruction document of Yatian LiDAR(V1.0)

## 以当前目录下的examples文件夹下的example.cpp为例，进行运行详解

    First, switch the path to the current path under the build folder, 
    and you can execute cmake.. && make -j2 command can generate the corresponding executable file under bin, 
    and then execute it

## 下面是基于V1.0版本的调用激光雷达的一些接口的函数的说明
    1.createUnitreeLidarReader()
    创建雷达(对象) (注:指针形式)
    2.创建或初始化雷达对象后可以使用initialize去执行初始化
    3.setLidarWorkingMode(模式)
    设置雷达的工作模式   工作模式有两种
    NORMAL（1）Configure the LiDAR to operate in normal mode.
    STANDBY（2）Configure the LiDAR to enter standby mode.
    
    4.runParse()
    这个里面记录着一些信息，需要辅助解析后进行判断使用
    有 NONE   // No valid message
    IMU         // An IMU message
    POINTCLOUD      // A cached PointCloud message
    RANGE       // A Range message
    AUXILIARY       // An Auxiliary message
    VERSION         // A Version info
    TIME        // A time sync info

    5.getVersionOfFirmware()
    6.getVersionOfSDK()
    获取雷达的版本号
    7.getDirtyPercentage()
    获取脏污百分比

    8.setLEDDisplayMode(灯表)
    设置LED灯
    另外还有好几种模式可供调节
      FORWARD_SLOW=2,       /* Function mode forward at slow. | */
      FORWARD_FAST=3,       /* Function mode forward at fast. | */
      REVERSE_SLOW=4,       /* Function mode reverse at slow. | */
      REVERSE_FAST=5,       /* Function mode reverse at fast. | */
      TRIPLE_FLIP=6,        /* Function mode triple flip. | */
      TRIPLE_BREATHING=7,   /* Function mode triple breathing. | */
      SIXSTAGE_BREATHING=8  /* Function mode six-stage breathing. | */



    以上函数的具体设定与使用过程可参见examples下的YT_V1.0.cpp文件

### python的执行方式说明
    用命令行生成可执行文件后运行在bin文件夹下的test.py程序
    通过调用C的可执行文件获取到的数值数据依次为:
    时间戳、ID、点云数量、线圈数、x、y、z、强度、反射时间、线圈编号
    最新版本的通过python调用激光雷达的教程见python_files下的YT_V1.0.py
    
##  关于python_files文件夹下的YT_V1.0.py文件的说明
    在执行这个python程序文件前一定要完成相关的C文件的编译链接与生成，得到一个可执行的c程序文件
    YT_V1.0.py的主要功能包括，调用C的可执行文件，获取到它的输出以去做进一步的处理
    对于检测到的点云当前对其划分了四个分段，最近的点为红色、次近的点为黄色、再远些的点为
    蓝色、最远的点为绿色的



# AntennaDetection

LineSegment为摄像头二维图像的导线分割，采用Unet，制作的数据集all_data1源自米国的一个高压线数据集，两个数据集链接为：百度网盘。目前模型训练效果不佳，测试集性能较差，主要考虑数据集的制作优化，损失函数的优化（特别是导线很细，考虑前景背景像素不均衡导致的长尾问题）

pyrplidar为思岚lidar的python语言API，目前测试了一下，可以成功连接激光雷达并进行一些基本数据的读取，后续仍有待研究。c++的SDK不在项目中，不怎么会用，希望python好用一点。。。

目前的问题
（1）A1 lidar只能扫描二维平面，导线又非常细，很可能把导线扫描成少数的几个点；
（2）思岚lidar只能通过数据线或网线与设备连接，不能通过局域网，因此如果激光雷达装吊臂上，基本要用开发板（没法和电脑直连），做图像处理的话，开发板还需要和服务器网络通讯，感觉不好做；
（3）TBC

# 10.23 Update
视觉部分没用，用的纯激光雷达。

直接运行simple_scan.py即可。激光雷达会不断进行扫描，一秒获取到成百上千的点【角度，距离，数据质量等信息】，这些点类似于一种时间序列，有先后顺序。目前的思路是模拟吊臂在空中的场景，因为天空的环境很简单，只有树枝和导线，所以监测到有高质量【质量不为0】且距离小于阈值【dist_min】且角度在预定义好的检测角度【angle_min<x<angle_max】的点云，就把它记录一下（第一个不告警，因为怕误检，等监测到若干个异常点再告警），此时记录这个点的角度供下一个点来进行比较，筛出一些异常情况（因为如果是天空中的场景，距离导线2-4米的地方，这个距离通常是要比导线之间距离大不少的，因此根据激光雷达1cm的分辨率和导线2cm的直径可知，一根导线大概转一圈能扫到1个点，且吊臂移动相对缓慢，因此这些点的角度并不会有特别大的变化）

当后面的异常点的角度在以上一个点的角度为中心的某个临域中【max(anomaly_angle-angle_thres, 0) <= scan.angle <= min(anomaly_angle+angle_thres, 360)】，此时让计数器+1，并更新这个角度供后面的点进行校对。当计数器大于某个阈值【max_allowed_cnt】，就进行告警。

以上采集点云数据的操作循环进行，当2秒内没有新的异常点出现是，便终止告警音乐的播放，并且刚才如果检测到异常点云但cnt并没达到告警阈值，将所有的信息重置。

目前的问题与考虑：1）室内测试效果还可以，有待室外测试，把距离拉的更长一点；2）这个算法检测到任何障碍物均告警，因此需要到时候安装合适的角度以及调整检测的参数（例如angle_min，angle_max，dist_min）等，或者可以进一步考虑如何将细的物体单独提取出来，忽略那些粗一点的物体。

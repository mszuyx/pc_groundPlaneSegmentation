# Run_based_segmentation
A ROS node to perform `ground plane fitting` for Realsense camera pointcloud.

![pic1](./pic/1.png)

```bib
@inproceedings{Zermas2017Fast,
  title={Fast segmentation of 3D point clouds: A paradigm on LiDAR data for autonomous vehicle applications},
  author={Zermas, Dimitris and Izzat, Izzat and Papanikolopoulos, Nikolaos},
  booktitle={IEEE International Conference on Robotics and Automation},
  year={2017},
}
```

## Requirement
* [PCL](https://github.com/PointCloudLibrary/pcl)
* [realsense-ros](https://github.com/IntelRealSense/realsense-ros)


## Run
```bash
$ catkin_make
$ rosrun pc_gps groundplanesegmentation
```


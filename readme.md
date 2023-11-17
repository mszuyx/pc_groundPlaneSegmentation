A ROS node to perform `ground plane fitting` for Realsense camera pointcloud.
![image](https://github.com/mszuyx/pc_groundPlaneSegmentation/assets/37651144/0d1806c7-028f-4787-9d31-e128e1ee3f98)

The algorithm was inspired by:
```bib
@inproceedings{Zermas2017Fast,
  title={Fast segmentation of 3D point clouds: A paradigm on LiDAR data for autonomous vehicle applications},
  author={Zermas, Dimitris and Izzat, Izzat and Papanikolopoulos, Nikolaos},
  booktitle={IEEE International Conference on Robotics and Automation},
  year={2017},
}
```
To cite this algorithm:
```bib
@article{chen2023self, title={A Self-Supervised Miniature One-Shot Texture Segmentation (MOSTS) Model for Real-Time Robot Navigation and Embedded Applications}, author={Chen, Yu and Rastogi, Chirag and Zhou, Zheyu and Norris, William R}, journal={arXiv preprint arXiv:2306.08814}, year={2023} }
```
## Requirement
* [PCL](https://github.com/PointCloudLibrary/pcl)
* [realsense-ros](https://github.com/IntelRealSense/realsense-ros)


## Run
```bash
$ catkin_make
$ rosrun pc_gps groundplanesegmentation
```


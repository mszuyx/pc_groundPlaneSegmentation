cmake_minimum_required(VERSION 2.8.3)
project(pc_gps)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  pcl_ros
  pcl_conversions
)

catkin_package(CATKIN_DEPENDS
	sensor_msgs
)

###########
## Build ##
###########

include_directories(
  ${catkin_INCLUDE_DIRS}
)

find_package(Eigen3 REQUIRED)

#SET(CMAKE_CXX_FLAGS "-std=c++11 -O2 -g -Wall ${CMAKE_CXX_FLAGS}")
SET(CMAKE_CXX_FLAGS "-std=c++17 -O2 -g -Wall ${CMAKE_CXX_FLAGS}")
SET(CMAKE_CXX_STANDARD 17)#14
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

#Ground Plane Fitter
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(pc_gps pc_gps.cpp)

target_include_directories(pc_gps PRIVATE
	${PCL_INCLUDE_DIRS}
)

target_link_libraries(pc_gps
	${catkin_LIBRARIES}
	${PCL_LIBRARIES}
)


// Import ROS lib
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Imu.h>
// Import PCL lib
#include <pcl_ros/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
// Import Eigen lib
#include <Eigen/Dense>
// Import message_filters lib
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

//#include <tf/transform_listener.h>
//#include <pcl_ros/transforms.h>

using namespace message_filters;
using namespace Eigen;

// Declare helper function
bool point_cmp(pcl::PointXYZ a, pcl::PointXYZ b){return a.y>b.y;}

// Declare pc variables
pcl::PointCloud<pcl::PointXYZ>::Ptr seeds_pc(new pcl::PointCloud<pcl::PointXYZ>());
pcl::PointCloud<pcl::PointXYZ>::Ptr ground_pc(new pcl::PointCloud<pcl::PointXYZ>());
pcl::PointCloud<pcl::PointXYZ>::Ptr not_ground_pc(new pcl::PointCloud<pcl::PointXYZ>());

class GroundPlaneSeg{
public:
    GroundPlaneSeg();
private:
    // Declare sub & pub
    ros::NodeHandle node_handle_;
    ros::Publisher ground_points_pub_;
    ros::Publisher groundless_points_pub_;
    //ros::Publisher groundless_flat_msg_pub_;
    //tf::TransformListener tfListener;

    // Declare ROS params
    double sensor_height_;
    int num_iter_;
    int num_lpr_;
    double th_seeds_;
    double th_dist_;
    double th_ceil_;
    double th_box_;
    double map_unit_size_;
    std::string base_FrameId;
    double radius_search_;
    int in_radius_;
    double std_th_;
    int mean_k_;
    double alpha;
    
    // Sync settings
    message_filters::Subscriber<sensor_msgs::PointCloud2> points_node_sub_;
    message_filters::Subscriber<sensor_msgs::Imu> imu_node_sub_;
    typedef sync_policies::ApproximateTime<sensor_msgs::PointCloud2, sensor_msgs::Imu> RSSyncPolicy;
    typedef Synchronizer<RSSyncPolicy> Sync;
    boost::shared_ptr<Sync> sync;

    // Declare functions
    void quaternionMultiplication(double p0, double p1, double p2, double p3, double q0, double q1, double q2, double q3, double& r0, double& r1, double& r2, double& r3);
    void rotateVectorByQuaternion(double x, double y, double z, double q0, double q1, double q2, double q3, double& vx, double& vy, double& vz);
    void quaternionToMatrix(double q0, double q1, double q2, double q3,  Affine3d& transform);
    void imuNormal_update(double p0, double p1, double p2, double p3);
    void findMean(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pc, Eigen::Vector3d& pc_mean);
    void estimate_plane_(void);
    void extract_initial_seeds_(const pcl::PointCloud<pcl::PointXYZ>::Ptr& p_sorted);
    void rs_pc_callback_ (const sensor_msgs::PointCloud2ConstPtr& input_cloud, const sensor_msgs::Imu::ConstPtr& imu_msg);

    // Model parameter for ground plane fitting
    // The ground plane model is: ax+by+cz+d=0, here normal:=[a,b,c], d=d, th_dist_d_ = threshold_dist - d 
    double d_;
    MatrixXd normal_imu{3,1};
    double th_dist_d_;
    double last_lpr;
};

GroundPlaneSeg::GroundPlaneSeg():node_handle_("~"){
    // Init ROS related
    ROS_INFO("Inititalizing Ground Plane Segmentation Node...");

    node_handle_.param("sensor_height", sensor_height_, 1.0);
    ROS_INFO("Sensor Height: %f", sensor_height_);

    node_handle_.param("num_iter", num_iter_, 3);
    ROS_INFO("Num of Iteration: %d", num_iter_);

    node_handle_.param("num_lpr", num_lpr_, 400);
    ROS_INFO("Num of LPR: %d", num_lpr_);

    node_handle_.param("th_seeds", th_seeds_, 0.5);
    ROS_INFO("Seeds Threshold: %f", th_seeds_);

    node_handle_.param("th_dist", th_dist_, 0.05);
    ROS_INFO("Distance Threshold: %f", th_dist_);

    node_handle_.param("th_ceil_", th_ceil_, 1.0);
    ROS_INFO("Ceiling Threshold: %f", th_ceil_);

    node_handle_.param("th_box_", th_box_, 7.0);
    ROS_INFO("Box Threshold: %f", th_box_);

    node_handle_.param("map_unit_size_", map_unit_size_, 0.15); 
    ROS_INFO("map_unit_size_: %f", map_unit_size_);

    node_handle_.param("radius_search_", radius_search_, 0.15);
    ROS_INFO("radius_search_: %f", radius_search_);

    node_handle_.param("in_radius_", in_radius_, -25);
    ROS_INFO("in_radius_: %d", in_radius_);

    node_handle_.param("std_th_", std_th_, 0.12);
    ROS_INFO("std_th_: %f", std_th_);

    node_handle_.param("mean_k_", mean_k_, -15);
    ROS_INFO("mean_k_: %d", mean_k_);

    node_handle_.param("alpha", alpha, 0.1);
    ROS_INFO("moving average factor alpha: %f", alpha);

    //node_handle_.param<std::string>("frame_id", base_FrameId, "/base_link");
    //ROS_INFO("robot base frame_id: %s", base_FrameId.c_str());

    // Subscribe to realsense topic
    points_node_sub_.subscribe(node_handle_, "/cloud_in", 1); //5
    imu_node_sub_.subscribe(node_handle_, "/imu/data", 1);  //100
    // ApproximateTime takes a queue size as its constructor argument, hence RSSyncPolicy(xx)
    sync.reset(new Sync(RSSyncPolicy(10), points_node_sub_, imu_node_sub_));   
    sync->registerCallback(boost::bind(&GroundPlaneSeg::rs_pc_callback_, this, _1, _2));

    // Publish Init
    std::string no_ground_topic, ground_topic;
    node_handle_.param<std::string>("no_ground_point_topic", no_ground_topic, "/points_no_ground");
    ROS_INFO("Not Ground Output Point Cloud: %s", no_ground_topic.c_str());
    groundless_points_pub_ = node_handle_.advertise<sensor_msgs::PointCloud2>(no_ground_topic, 1);
    node_handle_.param<std::string>("ground_point_topic", ground_topic, "/points_ground");
    ROS_INFO("Only Ground Output Point Cloud: %s", ground_topic.c_str());
    ground_points_pub_ = node_handle_.advertise<sensor_msgs::PointCloud2>(ground_topic, 1);

    // For demo===============================================================================================
    /*
    std::string no_ground_flat_topic;
    node_handle_.param<std::string>("no_ground_flat_topic", no_ground_flat_topic, "/no_ground_flat");
    ROS_INFO("Not Ground Output Point Cloud: %s", no_ground_flat_topic.c_str());
    groundless_flat_msg_pub_ = node_handle_.advertise<sensor_msgs::PointCloud2>(no_ground_flat_topic, 1);
    */
    // =======================================================================================================

    normal_imu << 0.0,1.0,0.0;
    last_lpr = sensor_height_;
}

void GroundPlaneSeg::quaternionMultiplication(double p0, double p1, double p2, double p3,
                                              double q0, double q1, double q2, double q3,
                                              double& r0, double& r1, double& r2, double& r3){
    // r = p q
    r0 = p0*q0 - p1*q1 - p2*q2 - p3*q3;
    r1 = p0*q1 + p1*q0 + p2*q3 - p3*q2;
    r2 = p0*q2 - p1*q3 + p2*q0 + p3*q1;
    r3 = p0*q3 + p1*q2 - p2*q1 + p3*q0;
}

void GroundPlaneSeg::rotateVectorByQuaternion(double x, double y, double z, 
                                              double q0, double q1, double q2, double q3, 
                                              double& vx, double& vy, double& vz){ 
    vx = (q0*q0 + q1*q1 - q2*q2 - q3*q3)*x + 2*(q1*q2 - q0*q3)*y + 2*(q1*q3 + q0*q2)*z;
    vy = 2*(q1*q2 + q0*q3)*x + (q0*q0 - q1*q1 + q2*q2 - q3*q3)*y + 2*(q2*q3 - q0*q1)*z;
    vz = 2*(q1*q3 - q0*q2)*x + 2*(q2*q3 + q0*q1)*y + (q0*q0 - q1*q1 - q2*q2 + q3*q3)*z;
}

void GroundPlaneSeg::quaternionToMatrix(double q0, double q1, double q2, double q3, Affine3d& transform){
    double t0 = 2 * (q0 * q1 + q2 * q3);
    double t1 = 1 - 2 * (q1 * q1 + q2 * q2);
    double roll = std::atan2(t0, t1);

    double t2 = 2 * (q0 * q2 - q3 * q1);
    if (t2 >= 1){t2 = 1.0;}
    else if (t2<= -1){t2 = -1.0;}
    double pitch = std::asin(t2);

    //double t3 = 2 * (q0 * q3 + q1 * q2);
    //double t4 = 1 - 2 * (q2 * q2 + q3 * q3);
    //double yaw = std::atan2(t3, t4);

    // axis defined in camera_depth_optical_frame
    transform = AngleAxisd(roll+1.5708, Vector3d::UnitX()) * AngleAxisd(pitch, Vector3d::UnitZ()); 
}

void GroundPlaneSeg::imuNormal_update(double q0, double q1, double q2, double q3){
    // Rotate quaternion into proper frame:
    double q0_, q1_, q2_, q3_; 
    double q0_tf=0, q1_tf=0, q2_tf=0, q3_tf=1; 
    quaternionMultiplication(q0, q1, q2, q3,
                            q0_tf, q1_tf, q2_tf, q3_tf,
                            q0_, q1_, q2_, q3_);
    double ux=0, uy=0, uz=1;
    double lx, ly, lz;
    rotateVectorByQuaternion(ux, uy, uz,
                            q0_, -q1_, -q2_, -q3_, 
                            lx, ly, lz);
                                            
    normal_imu << lx,ly,-lz;
}

void GroundPlaneSeg::findMean(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pc, Eigen::Vector3d& pc_mean){
    double x_sum = 0;
    double y_sum = 0;
    double z_sum = 0;
    int cnt = 0;
    for(size_t i=0;i<pc->points.size();i++){
        x_sum += double(pc->points[i].x);
        y_sum += double(pc->points[i].y);
        z_sum += double(pc->points[i].z);
        cnt++;
    }
    x_sum = cnt!=0?x_sum/cnt:0;
    y_sum = cnt!=0?y_sum/cnt:0;
    z_sum = cnt!=0?z_sum/cnt:0;
    pc_mean << x_sum,y_sum,z_sum;
}

void GroundPlaneSeg::estimate_plane_(void){
    // Create covarian matrix in single pass.
    Eigen::Vector3d seeds_mean;
    findMean(ground_pc, seeds_mean);

    // according to normal.T*[x,y,z] = -d
    // d_ = -(normal_imu.transpose()*seeds_mean)(0,0);

    d_ = -seeds_mean[1];

    // set distance threhold to `th_dist - d`
    th_dist_d_ = -(th_dist_ + d_);
    // update the equation parameters
}

void GroundPlaneSeg::extract_initial_seeds_(const pcl::PointCloud<pcl::PointXYZ>::Ptr& p_sorted){
    // LPR is the mean of low point representative
    double sum = 0;
    int cnt = 0;
    // Calculate the mean height value.
    for(size_t i=0;i<p_sorted->points.size() && cnt<num_lpr_;i++){
        sum += double(p_sorted->points[i].y);
        cnt++;
    }
    double lpr_height = cnt!=0?sum/cnt:0;// in case divide by 0
    lpr_height = (alpha*last_lpr) + ((1-alpha)*lpr_height);

    seeds_pc->clear();
    // iterate pointcloud, filter those height is less than lpr.height+th_seeds_
    for(size_t i=0;i<p_sorted->points.size();i++){
        if(double(p_sorted->points[i].y) > lpr_height + th_seeds_){
            seeds_pc->points.push_back(p_sorted->points[i]);
        }
    }
    last_lpr = lpr_height;
    // update seeds points
}

void GroundPlaneSeg::rs_pc_callback_ (const sensor_msgs::PointCloud2ConstPtr& input_cloud, const sensor_msgs::Imu::ConstPtr& imu_msg){
    //ROS_INFO("callback"); 

    double q0_in, q1_in, q2_in, q3_in; 
    q0_in=imu_msg->orientation.w;
    q1_in=imu_msg->orientation.x;
    q2_in=imu_msg->orientation.y;
    q3_in=imu_msg->orientation.z;
     
    // Update ground plane normal vector based on the assumption that the normal vector is parallel to the gravity vector                         
    //imuNormal_update(q0_in, q1_in, q2_in, q3_in);

    // 1.Convert pc to pcl::PointXYZ
    pcl::PCLPointCloud2::Ptr input_cloud_pcl (new pcl::PCLPointCloud2 ());
    pcl_conversions::toPCL(*input_cloud, *input_cloud_pcl);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_raw (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::fromPCLPointCloud2(*input_cloud_pcl, *cloud_raw);

    // 2.Apply voxel filter
    if (map_unit_size_>0){
        pcl::VoxelGrid<pcl::PointXYZ> sor;
        sor.setInputCloud (cloud_raw);
        sor.setLeafSize (float (0.3*map_unit_size_),float (map_unit_size_),float (0.3*map_unit_size_));
        sor.filter (*cloud_raw);
    }

    // Transform pointcloud into robot base_frame
    /*
    tf::StampedTransform sensorToRobotTf;
    try {
        tfListener.lookupTransform(base_FrameId, input_cloud->header.frame_id, input_cloud->header.stamp, sensorToRobotTf);
    } catch(tf::TransformException& ex){
        ROS_ERROR_STREAM( "Transform error of sensor data: " << ex.what() << ", quitting callback");
        return;
    }
    Eigen::Matrix4f sensorToRobot;
    pcl_ros::transformAsMatrix(sensorToRobotTf, sensorToRobot);
    pcl::transformPointCloud(*cloud_raw, *cloud_raw, sensorToRobot);
    */
     
    // 3.Transform pointcloud w.r.t IMU reading
    Affine3d transform;
    quaternionToMatrix(q0_in, q1_in, q2_in, q3_in, transform);
    // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_org (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::transformPointCloud (*cloud_raw, *cloud_raw, transform);

    // 4.Clip based on box threshold.
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::CropBox<pcl::PointXYZ> boxFilter;
    boxFilter.setMin(Eigen::Vector4f(-th_box_, -1.5, 0.1, 1.0));
    boxFilter.setMax(Eigen::Vector4f(th_box_, 2.0*sensor_height_, th_box_, 1.0));
    boxFilter.setInputCloud(cloud_raw);
    boxFilter.filter(*cloud);
    pcl::copyPointCloud<pcl::PointXYZ,pcl::PointXYZ>(*cloud, *cloud_raw);

    // 5.Apply radius removal filter
    if (in_radius_>0){
        pcl::RadiusOutlierRemoval<pcl::PointXYZ> outrem;
        outrem.setInputCloud(cloud);
        outrem.setRadiusSearch(radius_search_);
        outrem.setMinNeighborsInRadius (in_radius_);
        //outrem.setKeepOrganized(true);
        outrem.filter (*cloud);  
    }

    // 6.Apply statistical outlier removal
    if (mean_k_>0){
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud (cloud);
        sor.setMeanK (mean_k_);
        sor.setStddevMulThresh (std_th_);
        sor.filter (*cloud);
    }

    // 7.Sort on Y-axis value
    sort(cloud->points.begin(),cloud->end(),point_cmp);

    // 8. Extract init ground seeds
    extract_initial_seeds_(cloud);
    ground_pc = seeds_pc;

    // 9. Ground plane fitter mainloop
    for(int i=0;i<num_iter_;i++){
        estimate_plane_();
        // Clear memory
        ground_pc->clear();
        // Threshold filter
        if(i<num_iter_-1){
            for(size_t r=0;r<cloud->points.size();r++){
                if(double((*cloud)[r].y)>th_dist_d_){
                    ground_pc->points.push_back((*cloud)[r]);
                }
            } 
        }else{ // Reach last iteration
            not_ground_pc->clear();
            for(size_t r=0;r<cloud_raw->points.size();r++){
                if(double((*cloud_raw)[r].y)>th_dist_d_ && double((*cloud_raw)[r].y)<(th_dist_d_+(2*th_dist_))){
                    ground_pc->points.push_back((*cloud_raw)[r]);
                }else if(double((*cloud_raw)[r].y)<-th_ceil_){
                    continue;
                }else{
                    not_ground_pc->points.push_back((*cloud_raw)[r]);
                }
            }
        }
    }
 
    // publish ground points
    sensor_msgs::PointCloud2 ground_msg;
    pcl::toROSMsg(*ground_pc, ground_msg);
    ground_msg.header.stamp = input_cloud->header.stamp;
    ground_msg.header.frame_id = input_cloud->header.frame_id;
    ground_points_pub_.publish(ground_msg);
    //publish not ground points
    sensor_msgs::PointCloud2 groundless_msg;
    pcl::toROSMsg(*not_ground_pc, groundless_msg);
    groundless_msg.header.stamp = input_cloud->header.stamp;
    groundless_msg.header.frame_id = input_cloud->header.frame_id;
    groundless_points_pub_.publish(groundless_msg);

    // For demo===============================================================================================
    /*
    pcl::PointCloud<pcl::PointXYZ>::Ptr not_ground_flat (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::copyPointCloud<pcl::PointXYZ,pcl::PointXYZ>(*not_ground_pc, *not_ground_flat);
    for(size_t i=0;i<not_ground_flat->points.size();i++){
    (*not_ground_flat)[i].y = 0;
    }
    // publish not ground flat
    sensor_msgs::PointCloud2 groundless_flat_msg;
    pcl::toROSMsg(*not_ground_flat, groundless_flat_msg);
    groundless_flat_msg.header.stamp = input_cloud->header.stamp;
    groundless_flat_msg.header.frame_id = input_cloud->header.frame_id;
    groundless_flat_msg_pub_.publish(groundless_flat_msg);
    */
    // =======================================================================================================
}

int main (int argc, char** argv) {
    ros::init(argc, argv, "GroundPlaneSeg");
    GroundPlaneSeg node;
    // ros::MultiThreadedSpinner spinner(4); // Use 4 threads
    // spinner.spin();
    ros::spin();
    return 0;
 }

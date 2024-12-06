#include <string>
#include <boost/algorithm/string.hpp>

//ROS
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <octomap_msgs/Octomap.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <octomap/OcTree.h>
#include <octomap_msgs/conversions.h>

//PCL
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>

#define PRINTF_REGULAR "\x1B[0m"
#define PRINTF_RED "\x1B[31m"
#define PRINTF_GREEN "\x1B[32m"
#define PRINTF_YELLOW "\x1B[33m"
#define PRINTF_BLUE "\x1B[34m"
#define PRINTF_MAGENTA "\x1B[35m"
#define PRINTF_CYAN "\x1B[36m"
#define PRINTF_WHITE "\x1B[37m"

class Filtering{

    public:
        
        Filtering(float z_filter_min_, float z_filter_max_, float z_map_max_);
        void startFiltering(sensor_msgs::PointCloud2 in_cloud_);

        bool debug_rgs = true;
        float z_filter_min, z_filter_max, z_map_max; //values to filter the PointCloud in z axes

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_obstacles ;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_obstacles_negative;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_traversable;
};

Filtering::Filtering(float z_filter_min_, float z_filter_max_, float z_map_max_){
    z_filter_min = z_filter_min_; 
    z_filter_max = z_filter_max_; 
    z_map_max = z_map_max_;
}

void Filtering::startFiltering(sensor_msgs::PointCloud2 in_cloud_)
{
    ROS_INFO_COND(debug_rgs,PRINTF_CYAN"    Initializing Filtering Process");
    
    /* Stage 01: Create variables */
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>); // Create a pcl PC to save map received from PC2
    pcl::fromROSMsg(in_cloud_, *cloud); // Copy all the data from in_cloud to cloud

    /* Stage 02: Filters the point cloud using a PassThrough filter. */
    pcl::IndicesPtr indices (new std::vector <int>); // Smart pointer to store indices of points that pass through the filter
    pcl::PassThrough<pcl::PointXYZ> pass_through; // PassThrough filter, which limits points in the cloud based on a specific field
    pass_through.setInputCloud (cloud); // Specifies the input point cloud (cloud) to be filtered
    pass_through.setFilterFieldName ("z"); // The filter will operate on the z coordinate of the points.
    pass_through.setFilterLimits(z_filter_min, z_filter_max); // Filter limits
    // Is posibble to play with this limits to simplify the segmentation. For example knowing the z limit in travesable area.
    pass_through.filter (*indices); // Filters the input cloud and stores the indices of valid points in indices
    ROS_INFO_COND(debug_rgs,PRINTF_MAGENTA"     Filtering: Filtered Traversable PointCloud size %lu", indices->size());
    
    ROS_INFO_COND(debug_rgs,PRINTF_MAGENTA"     Filtering: Initializing filtering initial PointCloud");
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_point_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr non_filtered_point_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud);       // Original PointCloud
    extract.setIndices(indices);        // Valid indexes obtained from PassThrough
    extract.setNegative(true);          // Switch to positive mode for invalid points
    extract.filter(*filtered_point_cloud); // Points that did NOT meet the filter
    ROS_INFO_COND(debug_rgs,PRINTF_MAGENTA"     Filtering: Filtered PointCloud size %lu", filtered_point_cloud->size());
    pcl::IndicesPtr indices2 (new std::vector <int>); // Smart pointer to store indices of points that pass through the filter
    pcl::PassThrough<pcl::PointXYZ> pass_through2; // PassThrough filter, which limits points in the cloud based on a specific field
    pass_through2.setInputCloud (filtered_point_cloud); // Specifies the input point cloud (cloud) to be filtered
    pass_through2.setFilterFieldName ("z"); // The filter will operate on the z coordinate of the points.
    pass_through2.setFilterLimits(z_filter_max, z_map_max); // Filter limits
    pass_through2.filter (*indices2); // Filters the input cloud and stores the indices of valid points in indices
    cloud_obstacles = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
    cloud_obstacles->height = 1;
    cloud_obstacles->is_dense = false;
    cloud_obstacles->header.frame_id = cloud->header.frame_id;
    cloud_obstacles->width = indices2->size();
    ROS_INFO_COND(debug_rgs,PRINTF_MAGENTA"     Filtering: Filtered filtered upper PointCloud size %lu", indices2->size());
    for(const int& index : *indices2){ 
        cloud_obstacles->push_back(filtered_point_cloud->points[index]);             
    }
    pcl::ExtractIndices<pcl::PointXYZ> extract2;
    extract2.setInputCloud(cloud);       // Original PointCloud
    extract2.setIndices(indices);        // Valid indexes obtained from PassThrough
    extract2.setNegative(false);          // Switch to negative mode for valid points
    extract2.filter(*non_filtered_point_cloud); // Points that did NOT meet the filter
    cloud = non_filtered_point_cloud;
    ROS_INFO_COND(debug_rgs,PRINTF_MAGENTA"     Filtering: Finishing filtering initial PointCloud");

    pcl::PointCloud<pcl::PointXYZ>::Ptr non_filtered_point_cloud2(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::IndicesPtr indices3(new std::vector <int>); // Smart pointer to store indices of points that pass through the filter
    pcl::PassThrough<pcl::PointXYZ> pass_through3; // PassThrough filter, which limits points in the cloud based on a specific field
    pass_through3.setInputCloud (cloud); // Specifies the input point cloud (cloud) to be filtered
    pass_through3.setFilterFieldName ("z"); // The filter will operate on the z coordinate of the points.
    pass_through3.setFilterLimits(0.15, z_filter_max); // Filter limits
    pass_through3.filter (*indices3); // Filters the input cloud and stores the indices of valid points in indices
    cloud_traversable = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
    cloud_traversable->height = 1;
    cloud_traversable->is_dense = false;
    cloud_traversable->header.frame_id = cloud->header.frame_id;
    cloud_traversable->width = indices3->size();
    ROS_INFO_COND(debug_rgs,PRINTF_MAGENTA"     Filtering: Filtered filtered buttom PointCloud size %lu", indices3->size());
    for(const int& index : *indices3){ 
        cloud_traversable->push_back(cloud->points[index]);             
    }
    pcl::ExtractIndices<pcl::PointXYZ> extract3;
    extract3.setInputCloud(cloud);       // Original PointCloud
    extract3.setIndices(indices3);        // Valid indexes obtained from PassThrough
    extract3.setNegative(true);          // Switch to true mode for invalid points
    extract3.filter(*non_filtered_point_cloud2); // Points that did NOT meet the filter
    cloud_obstacles_negative = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
    cloud_obstacles_negative = non_filtered_point_cloud2;
    ROS_INFO_COND(debug_rgs,PRINTF_MAGENTA"     Filtering: Finishing filtering initial PointCloud");
}

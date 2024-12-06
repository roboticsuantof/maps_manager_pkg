#include <string>
#include <boost/algorithm/string.hpp>

//ROS
#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <tf2_ros/transform_listener.h>
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

std::string world_frame, ugv_base_frame;

ros::Publisher segmented_traversable_pc_pub, segmented_obstacles_pc_pub, filtered_pc_pub,reduced_map_pub;
ros::Subscriber new_octomap_sub;
sensor_msgs::PointCloud2 data_in;
std::shared_ptr<tf2_ros::Buffer> tfBuffer;
std::unique_ptr<tf2_ros::TransformListener> tf2_list;
std::unique_ptr<tf::TransformListener> tf_list_ptr;
std::string pc_sub;
geometry_msgs::Vector3 init_pose;

bool debug_rgs, latch_topic, use_tf_pose, merge_clusters_by_normals, filtered_PointCloud, delimit_PointCloud, apply_segmentation;
bool second_delimit_PointCloud;
int normal_estimator_search; // The number of neighboring points to consider when computing the normals. (50 initialy, 20 for constrained scenarios)
float z_filter_min, z_filter_max, z_map_max; //values to filter the PointCloud in z axes
int min_size_cluster, max_size_cluster; // values min and max to the posibles Point Cluster
int num_neighbours; // number of neighbours considered to create a cluster
float degree_smoothness_threshold, curvature_threshold;
float normal_threshold; 
std::vector<int> v_cluster_trav, v_cluster_obst;

pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg_grow;
pcl::PointXYZ searchPoint;

void startRegionGrowing(sensor_msgs::PointCloud2 in_cloud_);
void pcCallback(const sensor_msgs::PointCloud2::ConstPtr& msg);
void getPositionUGV();
void publishTopicsPointCloud();
void pointCloudToOctomap(sensor_msgs::PointCloud2 msg);
bool areNormalsSimilar(const Eigen::Vector3f& n1, const Eigen::Vector3f& n2, float threshold);

sensor_msgs::PointCloud2 pc_obstacles_out;
sensor_msgs::PointCloud2 pc_traversable_out;
sensor_msgs::PointCloud2 pc_filtered_out;
pcl::PointCloud<pcl::PointXYZ>::Ptr segmented_cloud_traversable (new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr segmented_cloud_obstacles (new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr negative_obstacles_PointCloud(new pcl::PointCloud<pcl::PointXYZ>);

const double res = 0.1;
float init_pose_x, init_pose_y, init_pose_z;

void pcCallback(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
    data_in = *msg;
    ROS_INFO_COND(debug_rgs,PRINTF_CYAN"    RegionGrowing:  pcCallback() --> data_in.size() = %i [%i / %i]",data_in.height *data_in.width, data_in.height , data_in.width);
	max_size_cluster = data_in.height *data_in.width; // The maximum cluster size is redifine
    startRegionGrowing(data_in);
}

void startRegionGrowing(sensor_msgs::PointCloud2 in_cloud_)
{
    ROS_INFO_COND(debug_rgs,PRINTF_CYAN"    RegionGrowing:  world_frame= %s  use_tf_pose,  ugv_base_frame= %s",world_frame.c_str() ,ugv_base_frame.c_str());
    
    /* Stage 01: Load robot initial position */
    getPositionUGV();   
	ROS_INFO_COND(debug_rgs,PRINTF_CYAN"    RegionGrowing:  Initial Pos: searchPoint=[%f %f %f]",searchPoint.x , searchPoint.y, searchPoint.z);

    /* Stage 02: Create variables */
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>); // Create a pcl PC to save map received from PC2
    pcl::fromROSMsg(in_cloud_, *cloud); // Copy all the data from in_cloud to cloud
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	Eigen::Vector3d init_point_;
    kdtree.setInputCloud (cloud); // Create kdtree with all cloud data

    /* Stage 03: Set kdtree parameters with K nearest neighbor from initial point search */
    int K = 20; // neighbor number to consider for each point examined
    std::vector<int> pointIdxNKNSearch(K);
    std::vector<float> pointNKNSquaredDistance(K);
    kdtree.nearestKSearch (searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance);
    init_point_.x() = cloud->points[pointIdxNKNSearch[0]].x;
	init_point_.y() = cloud->points[pointIdxNKNSearch[0]].y;
	init_point_.z() = cloud->points[pointIdxNKNSearch[0]].z;
    int id_nearest1_ = pointIdxNKNSearch[0];

    /* Stage 04: Filters the point cloud using a PassThrough filter. */
    pcl::IndicesPtr indices (new std::vector <int>); // Smart pointer to store indices of points that pass through the filter
    if(delimit_PointCloud){
        pcl::PassThrough<pcl::PointXYZ> pass_through; // PassThrough filter, which limits points in the cloud based on a specific field
        pass_through.setInputCloud (cloud); // Specifies the input point cloud (cloud) to be filtered
        pass_through.setFilterFieldName ("z"); // The filter will operate on the z coordinate of the points.
        pass_through.setFilterLimits(z_filter_min, z_filter_max); // Filter limits
        // Is posibble to play with this limits to simplify the segmentation. For example knowing the z limit in travesable area.
        pass_through.filter (*indices); // Filters the input cloud and stores the indices of valid points in indices
        ROS_INFO_COND(debug_rgs,PRINTF_MAGENTA"     RegionGrowing: Filtered Traversable PointCloud size %lu", indices->size());
    }
    
    if(filtered_PointCloud){
        ROS_INFO_COND(debug_rgs,PRINTF_MAGENTA"     RegionGrowing: Initializing filtering initial PointCloud");
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_point_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr non_filtered_point_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(cloud);       // Original PointCloud
        extract.setIndices(indices);        // Valid indexes obtained from PassThrough
        extract.setNegative(true);          // Switch to negative mode for valid points
        extract.filter(*filtered_point_cloud); // Points that did NOT meet the filter
        ROS_INFO_COND(debug_rgs,PRINTF_MAGENTA"     RegionGrowing: Filtered PointCloud size %lu", filtered_point_cloud->size());
        pcl::IndicesPtr indices2 (new std::vector <int>); // Smart pointer to store indices of points that pass through the filter
        pcl::PassThrough<pcl::PointXYZ> pass_through2; // PassThrough filter, which limits points in the cloud based on a specific field
        pass_through2.setInputCloud (filtered_point_cloud); // Specifies the input point cloud (cloud) to be filtered
        pass_through2.setFilterFieldName ("z"); // The filter will operate on the z coordinate of the points.
        pass_through2.setFilterLimits(z_filter_max, z_map_max); // Filter limits
        pass_through2.filter (*indices2); // Filters the input cloud and stores the indices of valid points in indices
        segmented_cloud_obstacles->height = 1;
        segmented_cloud_obstacles->is_dense = false;
        segmented_cloud_obstacles->header.frame_id = cloud->header.frame_id;
        segmented_cloud_obstacles->width = indices2->size();
        ROS_INFO_COND(debug_rgs,PRINTF_MAGENTA"     RegionGrowing: Filtered filtered upper PointCloud size %lu", indices2->size());
        for(const int& index : *indices2){ 
            segmented_cloud_obstacles->push_back(filtered_point_cloud->points[index]);             
        }
        pcl::ExtractIndices<pcl::PointXYZ> extract2;
        extract2.setInputCloud(cloud);       // Original PointCloud
        extract2.setIndices(indices);        // Valid indexes obtained from PassThrough
        extract2.setNegative(false);          // Switch to negative mode for invalid points
        extract2.filter(*non_filtered_point_cloud); // Points that did NOT meet the filter
        cloud = non_filtered_point_cloud;
        ROS_INFO_COND(debug_rgs,PRINTF_MAGENTA"     RegionGrowing: Finishing filtering initial PointCloud");

        if(second_delimit_PointCloud){
        pcl::PointCloud<pcl::PointXYZ>::Ptr non_filtered_point_cloud2(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::IndicesPtr indices3(new std::vector <int>); // Smart pointer to store indices of points that pass through the filter
            pcl::PassThrough<pcl::PointXYZ> pass_through3; // PassThrough filter, which limits points in the cloud based on a specific field
            pass_through3.setInputCloud (cloud); // Specifies the input point cloud (cloud) to be filtered
            pass_through3.setFilterFieldName ("z"); // The filter will operate on the z coordinate of the points.
            pass_through3.setFilterLimits(0.15, z_filter_max); // Filter limits
            pass_through3.filter (*indices3); // Filters the input cloud and stores the indices of valid points in indices
            segmented_cloud_traversable->height = 1;
            segmented_cloud_traversable->is_dense = false;
            segmented_cloud_traversable->header.frame_id = cloud->header.frame_id;
            segmented_cloud_traversable->width = indices3->size();
            ROS_INFO_COND(debug_rgs,PRINTF_MAGENTA"     RegionGrowing: Filtered filtered buttom PointCloud size %lu", indices3->size());
            for(const int& index : *indices3){ 
                segmented_cloud_traversable->push_back(cloud->points[index]);             
            }
            pcl::ExtractIndices<pcl::PointXYZ> extract3;
            extract3.setInputCloud(cloud);       // Original PointCloud
            extract3.setIndices(indices3);        // Valid indexes obtained from PassThrough
            extract3.setNegative(true);          // Switch to true mode for invalid points
            extract3.filter(*non_filtered_point_cloud2); // Points that did NOT meet the filter
            negative_obstacles_PointCloud = non_filtered_point_cloud2;
            cloud = segmented_cloud_traversable;
            ROS_INFO_COND(debug_rgs,PRINTF_MAGENTA"     RegionGrowing: Finishing filtering initial PointCloud");
        }
    }
   
    publishTopicsPointCloud();
}

bool areNormalsSimilar(const Eigen::Vector3f& n1, const Eigen::Vector3f& n2, float threshold) {
    // Normalization of the vectors
    Eigen::Vector3f n1_normalized = n1.normalized();
    Eigen::Vector3f n2_normalized = n2.normalized();

    float cos_theta = n1_normalized.dot(n2_normalized); // Compute COS of the angle
    // std::cout << "          Between normals: cos(tetha)= " << cos_theta << "/"<< threshold <<" , theta= " << acos(cos_theta) << "";

    // Compare whit the threshold
    if(cos_theta > threshold)
        return true;
    else
        return false;
}

void publishTopicsPointCloud()
{
    ROS_INFO_COND(debug_rgs,PRINTF_CYAN"    RegionGrowing:  segmented_cloud_obstacles->size() = %lu", segmented_cloud_obstacles->size());
    pcl::toROSMsg(*segmented_cloud_obstacles, pc_obstacles_out);
    segmented_obstacles_pc_pub.publish(pc_obstacles_out);

    ROS_INFO_COND(debug_rgs,PRINTF_CYAN"    RegionGrowing:  segmented_cloud_traversable->size() = %lu", segmented_cloud_traversable->size());
    pcl::toROSMsg(*segmented_cloud_traversable, pc_traversable_out);
    segmented_traversable_pc_pub.publish(pc_traversable_out);

    if(filtered_PointCloud){
        ROS_INFO_COND(debug_rgs,PRINTF_CYAN"    RegionGrowing:  filtered_cloud->size() = %lu", negative_obstacles_PointCloud->size());
        pcl::toROSMsg(*negative_obstacles_PointCloud, pc_filtered_out);
        filtered_pc_pub.publish(pc_filtered_out);
    }

    ROS_INFO_COND(debug_rgs,PRINTF_CYAN"    \n... PointCloud published successfully , check publicated topics...");
}

void pointCloudToOctomap(sensor_msgs::PointCloud2 msg)
{
	//********* Retirive and process raw pointcloud************
	std::cout<<"Recieved cloud"<<std::endl;
	std::cout<<"Create Octomap"<<std::endl;
	octomap::OcTree tree(res);
	std::cout<<"Load points "<<std::endl;
	pcl::PCLPointCloud2 cloud;
	pcl_conversions::toPCL(msg,cloud);
	pcl::PointCloud<pcl::PointXYZ> pcl_pc;
    pcl::fromPCLPointCloud2(cloud, pcl_pc);
    std::cout<<"Filter point clouds for NAN"<<std::endl;
	std::vector<int> nan_indices;
	pcl::removeNaNFromPointCloud(pcl_pc,pcl_pc,nan_indices);
	octomap::Pointcloud oct_pc;
	octomap::point3d origin(0.0f,0.0f,0.0f);
	std::cout<<"Adding point cloud to octomap"<<std::endl;
	//octomap::point3d origin(0.0f,0.0f,0.0f);
	for(int i = 0;i<pcl_pc.points.size();i++){
		oct_pc.push_back((float) pcl_pc.points[i].x,(float) pcl_pc.points[i].y,(float) pcl_pc.points[i].z);
    }
	tree.insertPointCloud(oct_pc,origin,-1,false,false);
	
	std::cout<<"finished"<<std::endl;
	std::cout<<std::endl;

    octomap_msgs::Octomap octomap_reduced;
    octomap_reduced.binary = false;
    octomap_reduced.id = 1 ;
    octomap_reduced.resolution =0.1;
    octomap_reduced.header.frame_id = "/map";
    octomap_reduced.header.stamp = ros::Time::now();
    octomap_msgs::fullMapToMsg(tree, octomap_reduced);
    reduced_map_pub.publish(octomap_reduced);
}

void getPositionUGV()
{
    geometry_msgs::TransformStamped ret;
    if(use_tf_pose){
        try
        {
            ret = tfBuffer->lookupTransform(world_frame, ugv_base_frame, ros::Time(0));
            ROS_INFO_COND(debug_rgs,PRINTF_CYAN"    RegionGrowing:  getPositionUGV() :  Got lookupTransform ");
        }
        catch (tf2::TransformException &ex)
        {
            ROS_WARN("  RegionGrowing:  Region Growing Segmentation: Couldn't get UGV Pose (frame: %s), so not possible to set UGV start point; tf exception: %s", ugv_base_frame.c_str(),ex.what());
        }
        searchPoint.x = ret.transform.translation.x;
        searchPoint.y = ret.transform.translation.y;
        searchPoint.z = ret.transform.translation.z;
    }
    else{
        searchPoint.x = init_pose.x;
        searchPoint.y = init_pose.y;
        searchPoint.z = init_pose.z;
    }
	ROS_INFO_COND(debug_rgs,PRINTF_CYAN"    RegionGrowing: getPositionUGV() --> Initial Pos [%f %f %f]",searchPoint.x, searchPoint.y, searchPoint.z);
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "region_growing_segmentation_node");
	ros::NodeHandle n;
	ros::NodeHandle pn("~");

    ROS_INFO(PRINTF_CYAN"   RegionGrowing:  Segmentation: Node Initialed!!");
  
	pn.param<std::string>("world_frame",  world_frame, "map");
  	pn.param<std::string>("ugv_base_frame", ugv_base_frame, "ugv_base_link");
  	pn.param<std::string>("pc_sub", pc_sub, "/octomap_point_cloud_centers");
  	pn.param<bool>("debug_rgs", debug_rgs, true);
  	pn.param<bool>("latch_topic", latch_topic, true);
  	pn.param<bool>("use_tf_pose", use_tf_pose, true);
  	pn.param<bool>("merge_clusters_by_normals", merge_clusters_by_normals, false);
  	pn.param<bool>("delimit_PointCloud", delimit_PointCloud, true);
  	pn.param<bool>("second_delimit_PointCloud", second_delimit_PointCloud, false);
  	pn.param<bool>("filtered_PointCloud", filtered_PointCloud, false);
  	pn.param<bool>("apply_segmentation", apply_segmentation, true);

  	pn.param<float>("init_pose_x", init_pose_x, 0.0);
  	pn.param<float>("init_pose_y", init_pose_y, 0.0);
  	pn.param<float>("init_pose_z", init_pose_z, 0.0);

  	pn.param<int>("normal_estimator_search", normal_estimator_search, 50);
  	pn.param<float>("z_filter_min", z_filter_min, -std::numeric_limits<float>::max());
  	pn.param<float>("z_filter_max", z_filter_max, std::numeric_limits<float>::max());
  	pn.param<float>("z_map_max", z_map_max, std::numeric_limits<float>::max());
  	pn.param<int>("min_size_cluster", min_size_cluster, 50);
  	pn.param<int>("max_size_cluster", max_size_cluster, 1000000); // This value is refine above
  	pn.param<int>("num_neighbours", num_neighbours, 60);
  	pn.param<float>("degree_smoothness_threshold", degree_smoothness_threshold, 4.0);
  	pn.param<float>("curvature_threshold", curvature_threshold, 10.0);
  	pn.param<float>("normal_threshold", normal_threshold, 0.95);
	
    tfBuffer.reset(new tf2_ros::Buffer);
	tf2_list.reset(new tf2_ros::TransformListener(*tfBuffer));
	tf_list_ptr.reset(new tf::TransformListener(ros::Duration(5)));

    new_octomap_sub = n.subscribe<sensor_msgs::PointCloud2>(pc_sub, 1000, pcCallback);

	segmented_traversable_pc_pub = n.advertise<sensor_msgs::PointCloud2>("region_growing_traversability_pc_map", 10,latch_topic);
	segmented_obstacles_pc_pub = n.advertise<sensor_msgs::PointCloud2>("region_growing_obstacles_pos_pc_map", 10,latch_topic);
	filtered_pc_pub = n.advertise<sensor_msgs::PointCloud2>("region_growing_obstacles_neg_pc_map", 10,latch_topic);
    reduced_map_pub = n.advertise<octomap_msgs::Octomap>("region_growing_octomap_reduced", 100, latch_topic);

    init_pose.x = init_pose_x;
    init_pose.y = init_pose_y;
    init_pose.z = init_pose_z;
    
	while (ros::ok()) {
  
        ros::spin();
    }	
	
	return 0;
}

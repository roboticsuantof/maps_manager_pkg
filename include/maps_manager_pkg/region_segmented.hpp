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

class Segmentation
{
	public:
        Segmentation(float z_filter_min_, float z_map_max_, float px_, float py_, float pz_, bool merged_);
        void config(int N_estim, int min_cluster, int max_cluster, int n_neighbours, float d_smoothness, float c_threshold, float N_threshold);
        void startRegionGrowing(sensor_msgs::PointCloud2 in_cloud_);
        bool areNormalsSimilar(const Eigen::Vector3f& n1, const Eigen::Vector3f& n2, float threshold);

        bool debug_rgs = true;
        bool merge_clusters_by_normals;
        int normal_estimator_search; // The number of neighboring points to consider when computing the normals. (50 initialy, 20 for constrained scenarios)
        int min_size_cluster, max_size_cluster; // values min and max to the posibles Point Cluster
        int num_neighbours; // number of neighbours considered to create a cluster
        float z_filter_min, z_filter_max, z_map_max; //values to filter the PointCloud in z axes
        float degree_smoothness_threshold, curvature_threshold;
        float normal_threshold; 
        std::vector<int> v_cluster_trav, v_cluster_obst;

        pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg_grow;
        pcl::PointXYZ searchPoint;

        sensor_msgs::PointCloud2 pc_obstacles_out;
        sensor_msgs::PointCloud2 pc_traversable_out;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_traversable;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_obstacles;
};

Segmentation::Segmentation(float z_filter_min_, float z_map_max_, float px_, float py_, float pz_, bool merged_)
{
    z_filter_min = z_filter_min_;
    z_map_max = z_map_max_;
    searchPoint.x = px_;
    searchPoint.y = py_;
    searchPoint.z = pz_;
    merge_clusters_by_normals = merged_;
}

void Segmentation::config(int N_estim, int min_cluster, int max_cluster, int n_neighbours, float d_smoothness, float c_threshold, float N_threshold){
    normal_estimator_search = N_estim; 
    min_size_cluster = min_cluster, 
    max_size_cluster = max_cluster; 
    num_neighbours = n_neighbours; 
    degree_smoothness_threshold = d_smoothness; 
    curvature_threshold = c_threshold;
    normal_threshold = N_threshold; 
}

void Segmentation::startRegionGrowing(sensor_msgs::PointCloud2 in_cloud_)
{
    /* Stage 01: Load robot initial position */
	ROS_INFO_COND(debug_rgs,PRINTF_CYAN"    Segmentation:  Initial Pos: searchPoint=[%f %f %f]",searchPoint.x , searchPoint.y, searchPoint.z);
	ROS_INFO_COND(debug_rgs,PRINTF_CYAN"    Segmentation:  normal_estimator_search= %i , min_size_cluster= %i , max_size_cluster=% i , num_neighbours = %i, \ndegree_smoothness_threshold = %f, curvature_threshold =%f, normal_threshold = %f"
    ,normal_estimator_search, min_size_cluster, max_size_cluster, num_neighbours, degree_smoothness_threshold, curvature_threshold, normal_threshold);
    /* Stage 02: Create variables */
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>); // Create a pcl PC to save map received from PC2
    pcl::fromROSMsg(in_cloud_, *cloud); // Copy all the data from in_cloud to cloud

    /* Stage 03: Delimt the point cloud using a PassThrough filter. */
    pcl::IndicesPtr indices (new std::vector <int>); // Smart pointer to store indices of points that pass through the filter
    pcl::PassThrough<pcl::PointXYZ> pass_through; // PassThrough filter, which limits points in the cloud based on a specific field
    pass_through.setInputCloud (cloud); // Specifies the input point cloud (cloud) to be filtered
    pass_through.setFilterFieldName ("z"); // The filter will operate on the z coordinate of the points.
    pass_through.setFilterLimits(z_filter_min, z_map_max); // Filter limits
    // Is posibble to play with this limits to simplify the segmentation. For example knowing the z limit in travesable area.
    pass_through.filter (*indices); // Filters the input cloud and stores the indices of valid points in indices
    ROS_INFO_COND(debug_rgs,PRINTF_MAGENTA"     Segmentation: Initial PointCloud delimited size %lu", indices->size());
    
    ROS_INFO_COND(debug_rgs,PRINTF_MAGENTA"     Segmentation: Initializing filtering initial PointCloud");
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_point_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr non_filtered_point_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud);       // Original PointCloud
    extract.setIndices(indices);        // Valid indexes obtained from PassThrough
    extract.setNegative(false);          // Switch to negative mode for valid points
    extract.filter(*filtered_point_cloud); // Points that did NOT meet the filter
    cloud = filtered_point_cloud;
    ROS_INFO_COND(debug_rgs,PRINTF_MAGENTA"     Segmentation: Filtered PointCloud size %lu", cloud->size());

    /* Stage 04: Create Kdtree*/
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	Eigen::Vector3d init_point_;
    kdtree.setInputCloud (cloud); // Create kdtree with all cloud data

    /* Stage 05: Set kdtree parameters with K nearest neighbor from initial point search */
    int K = 20; // neighbor number to consider for each point examined
    std::vector<int> pointIdxNKNSearch(K);
    std::vector<float> pointNKNSquaredDistance(K);
    kdtree.nearestKSearch (searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance);
    init_point_.x() = cloud->points[pointIdxNKNSearch[0]].x;
	init_point_.y() = cloud->points[pointIdxNKNSearch[0]].y;
	init_point_.z() = cloud->points[pointIdxNKNSearch[0]].z;
    // int id_nearest1_ = pointIdxNKNSearch[0];   

    /* Stage 05: Estimating the normals of a 3D point cloud using the PCL library */
    ROS_INFO_COND(debug_rgs,PRINTF_CYAN"    Segmentation:  Starting Normal Estimation");
    pcl::search::Search<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>); // k-d tree search structure for efficient nearest-neighbor queries. 
    pcl::PointCloud <pcl::Normal>::Ptr normals (new pcl::PointCloud <pcl::Normal>); // Container to store the computed normals
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator; // Normal Estimation object, which calculates the normals for a point cloud
    normal_estimator.setSearchMethod (tree); // This allows the normal estimator to efficiently query neighboring points during computation.
    normal_estimator.setInputCloud (cloud); // Specifies the input point cloud (cloud) for which normals are to be calculated.
    normal_estimator.setKSearch (normal_estimator_search); // Configures the estimator to use the nearest neighbors (normal_estimator_search) for normal computation. A larger value smooths the normals but may reduce sensitivity to local variations.
    normal_estimator.compute (*normals); //  Performs the actual normal computation and stores the results in the normals point cloud.
    // For each point in the cloud:
        // - Query the 50 nearest neighbors using the k-d tree.
        // - Compute the surface normal by fitting a plane to the neighboring points using PCA (Principal Component Analysis).
        // - Store the computed normal and curvature in the normals point cloud.
    // The normals point cloud contains:
        // Normals: A 3D vector for each point, indicating the perpendicular direction to the local surface.
        // Curvature: A scalar value representing how sharply the surface bends at each point.
    ROS_INFO_COND(debug_rgs,PRINTF_CYAN"    Segmentation:  Finishing Normal Estimation");

    /* Stage 06: Performs Region Growing Segmentation to cluster the filtered index.*/
    ROS_INFO_COND(debug_rgs,PRINTF_CYAN"    Segmentation:  Starting Clusterization");
    reg_grow.setMinClusterSize (min_size_cluster); //  Clusters with fewer than min_size_cluster points are ignored.
    reg_grow.setMaxClusterSize (max_size_cluster);
    reg_grow.setSearchMethod (tree); // Uses the previously created k-d tree (tree) for efficient neighbor searches.
    reg_grow.setNumberOfNeighbours (num_neighbours); // Each point will consider its "num_neighbours" nearest neighbors for clustering.
    reg_grow.setInputCloud (cloud); //Specifies the filtered point cloud.
        // reg_grow.setIndices (indices);
    reg_grow.setInputNormals (normals); // Uses precomputed normals for each point to guide segmentation.
    reg_grow.setSmoothnessThreshold (degree_smoothness_threshold / 180.0 * M_PI); // Controls how much angular variation between neighboring points' normals is allowed within a cluster.
    reg_grow.setCurvatureThreshold (curvature_threshold); // Clusters are also constrained by the curvature of the points. Points with curvature greater than the threshold are excluded from the current cluster and treated as separate.

    std::vector <pcl::PointIndices> clusters; // Vector where each element contains the index of points belonging to one cluster.
    reg_grow.extract (clusters);
    ROS_INFO_COND(debug_rgs,PRINTF_CYAN"    Segmentation:  Finishing clusterization");

    ROS_INFO_COND(debug_rgs,PRINTF_CYAN"    Segmentation: Number of clusters is equal to: %lu ", clusters.size() );
    ROS_INFO_COND(debug_rgs,PRINTF_CYAN"    Segmentation: These are the indices of the points of the initial");
	
    /* Stage 07: Iterates over clusters generated by the region growing algorithm to compute the mean normals for each cluster*/
    std::vector<Eigen::Vector3f> mean_normals(clusters.size(), Eigen::Vector3f({ 0.,0.,0. })); // Creates a vector to store the average normal vector for each cluster.
        // The vector size is equal to the number of clusters (clusters.size())
        // Initialization: Each element is initialized to a zero vector ({0., 0., 0.})
    int cluster_idx_key = -1; // Is a variable to store the index of the cluster associated with the initial point (init_point_).
    int count = 0;
    for (size_t cluster_idx = 0 ; cluster_idx< clusters.size() ; cluster_idx ++)
    {
        for (int i= 0; i < clusters[cluster_idx].indices.size(); i++){  
            int index_ = clusters[cluster_idx].indices[i];
            if (cloud->points[index_].x == init_point_.x() && cloud->points[index_].y == init_point_.y() && cloud->points[index_].z == init_point_.z()){
                cluster_idx_key = cluster_idx;
                ROS_INFO_COND(debug_rgs,PRINTF_MAGENTA"     Segmentation: Number of initial traversability cluster: %i.", cluster_idx_key);
                ROS_INFO_COND(debug_rgs,PRINTF_MAGENTA"     Segmentation: Numer elements Point Cloud that belong to the first cluster: %lu .", clusters[cluster_idx_key].indices.size());
            }
            mean_normals[cluster_idx] += normals->points[i].getNormalVector3fMap(); //Accumulates the normal vector of the current point.
            count++;
        }
        mean_normals[cluster_idx].normalize(); // Normalizes the accumulated normal vector to compute the mean normal for the cluster.
        // std::cout << "cluster_idx: " << cluster_idx 
        //         << " , mean_normals[cluster_idx].normalize(): ["
        //         << mean_normals[cluster_idx].x()<< " , " 
        //         << mean_normals[cluster_idx].y()<< " , "  
        //         << mean_normals[cluster_idx].z()<< "] ,"
        //         << " cluster.size()= " << clusters[cluster_idx].indices.size()
        // << std:: endl;
    }
    ROS_INFO_COND(debug_rgs,PRINTF_MAGENTA"     Segmentation: Total Point saved in Clusters %i", count);

    /* Stage 08: Process to merge all the clouster with similar normal to cluster with inition position*/
    v_cluster_trav.clear(); v_cluster_obst.clear();
    v_cluster_trav.push_back(cluster_idx_key);
    int size_cluster_trav, size_cluster_obst;
    size_cluster_trav = size_cluster_obst = 0;
    size_cluster_trav = clusters[cluster_idx_key].indices.size();
    for (size_t cluster_idx = 0 ; cluster_idx< clusters.size() ; cluster_idx ++)
    {
        if(cluster_idx != cluster_idx_key ){
            if (areNormalsSimilar(mean_normals[cluster_idx_key], mean_normals[cluster_idx], normal_threshold)){
                // std::cout << " , cluster " << cluster_idx_key << " and " << cluster_idx << std::endl;
                v_cluster_trav.push_back(cluster_idx);
                size_cluster_trav = size_cluster_trav + clusters[cluster_idx].indices.size();
            }
            else{
                // std::cout << std::endl;
                v_cluster_obst.push_back(cluster_idx);
                size_cluster_obst = size_cluster_obst + clusters[cluster_idx].indices.size();
            }
        }
    }
    ROS_INFO_COND(debug_rgs,PRINTF_CYAN"    Segmentation: Clusters_Trav= %lu , Clusters_Obst= %lu ", v_cluster_trav.size(), v_cluster_obst.size() );

    /* Stage 09: Processes clusters of points obtained from segmentation, separates them into traversable and obstacle point clouds*/
    // Fill in the cloud data
    cloud_traversable = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
    cloud_obstacles = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);

    cloud_traversable->height = 1;
    cloud_traversable->is_dense = false; // Set to false because the cloud might contain invalid points.
    cloud_traversable->header.frame_id = cloud->header.frame_id;
    cloud_obstacles->height = 1;
    cloud_obstacles->is_dense = false;
    cloud_obstacles->header.frame_id = cloud->header.frame_id;

    if(merge_clusters_by_normals){
        cloud_traversable->width = size_cluster_trav;
        for(int i = 0; i < v_cluster_trav.size(); i++){ 
            // std::cout << "v_cluster_trav[i]= " << v_cluster_trav[i] << std::endl;
            int num_cluster_ = v_cluster_trav[i];
            for(int j = 0; j < clusters[num_cluster_].indices.size(); j++)
            {
                int index_ = clusters[num_cluster_].indices[j];
                cloud_traversable->push_back (cloud->points[index_]);             
            }
        }
        cloud_obstacles->width = size_cluster_obst;
        for(int i = 0; i < v_cluster_obst.size(); i++){ 
            // std::cout << "v_cluster_obst[i]= " << v_cluster_obst[i] << std::endl;
            int num_cluster_ = v_cluster_obst[i];
            for(int j = 0; j < clusters[num_cluster_].indices.size(); j++)
            {
                int index_ = clusters[num_cluster_].indices[j];
                cloud_obstacles->push_back (cloud->points[index_]);             
            }
        }
    }
    else{
        cloud_traversable->width = clusters[cluster_idx_key].indices.size();
        for (int i = 0; i < clusters[cluster_idx_key].indices.size(); i++)
        {
            int index_ = clusters[cluster_idx_key].indices[i];
            cloud_traversable->push_back (cloud->points[index_]);             
        }
        cloud_obstacles->width = (cloud->size() - clusters[cluster_idx_key].indices.size());
        for (size_t cluster_idx = 0 ; cluster_idx< clusters.size() ; cluster_idx ++)
        {
            if(cluster_idx_key != cluster_idx){
                for (int i = 0; i < clusters[cluster_idx].indices.size(); i++){  
                    int index_ = clusters[cluster_idx].indices[i];
                    cloud_obstacles->push_back (cloud->points[index_]);    
                }
            }
        }
    }
}

bool Segmentation::areNormalsSimilar(const Eigen::Vector3f& n1, const Eigen::Vector3f& n2, float threshold) {
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


# General Description maps_manager_pkg

This package provide tools to manage maps as launch for octomap, maps from bags, aloam for mapping task and segmentation PointCloud



## Use: Segmentate PountCloud as traversable and obstacles

The fallowing steps are create your Point Cloud segmentate. Is important to know that the initial Poincloud is segmentate in other two: Traversable PointCloud and Obstacles PointCloud. You must fallow four steps.

### First step

First is necessary to load the .bt file from the maps which is requiared to segmentate. For that we use octomap package.

    roslaunch maps_manager_pkg octomap_server.launch

### Second step

This step in to segmented the full point cloud. Is very important define correctly the initial point, which is considerer as the reference point to segmentate the traversable point cloud.

Here is possible to use the topic "clicked_point"  from rviz to identyfied whci point will be ours initial reference.

    roslaunch maps_manager_pkg region_growing.launch

### Third step

Convert the two PointCloud (traversable and obstacles) in two octomap .

    roslaunch maps_manager_pkg octomap_mapping_segmentation.launch

### Fourth step

Save the octomap as .bt file .

    roslaunch maps_manager_pkg octomap_saver_segmentation.launch
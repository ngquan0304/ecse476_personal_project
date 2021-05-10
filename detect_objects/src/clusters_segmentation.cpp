//* author: quan_nguyen
//* name: clusters_segmentation
//* purpose: this rosnode is a service that will segment a pointcloud to different meaningful clusters which are suppose to be blocks (with different shape) on a predefined table.

#include <ros/ros.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>

#include <std_srvs/Trigger.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h> //to convert between PCL and ROS
#include <pcl/conversions.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <pcl/common/common_headers.h>
#include <pcl/point_cloud.h>
#include <pcl/PCLHeader.h>

#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>

#include <pcl_utils/pcl_utils.h> 
#include <xform_utils/xform_utils.h>

#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Eigenvalues>
#include <eigen3/Eigen/SVD>

#include <detect_objects/ClusterSrvMsg.h>
#include <detect_objects/ClustersSegmentationTrigger.h>

using namespace std;
ros::NodeHandle* nh_ptr;
bool found_pcl_snapshot = false; //snapshot indicator
bool got_called = false;

pcl::PointCloud<pcl::PointXYZRGB>::Ptr input_cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>); //pointer for color version of pointcloud
sensor_msgs::PointCloud2 input_pcl, input_wrt_table_pcl, inbound_input_wrt_table_pcl, blocks_pcl, filtered_blocks_pcl, clusters_pcl, cluster_pcl;   // for debugging
std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> cluster_cloud_ptr_vec;

//* A call back function to get a snapshot of pointcloud
void cameraCB(const sensor_msgs::PointCloud2ConstPtr& cloud) {
    if (!found_pcl_snapshot) { // once only, to keep the data stable
        ROS_INFO("got new selected kinect image");
        pcl::fromROSMsg(*cloud, *input_cloud_ptr);
        // change header frame to the one we use
        input_cloud_ptr->header.frame_id = "camera_rgb_optical_frame"; // originally was depth one, not rgb
        ROS_INFO("image has  %d * %d points", input_cloud_ptr->width, input_cloud_ptr->height);
        found_pcl_snapshot = true;
    }
}

//* HSV to RGB helper
vector<int> HSVtoRGB(float H, float S,float V){
    if(H>360 || H<0 || S>100 || S<0 || V>100 || V<0){
        cout<<"The givem HSV values are not in valid range"<<endl;
        return vector<int> {0,0,0};
    }
    float s = S/100;
    float v = V/100;
    float C = s*v;
    float X = C*(1-abs(fmod(H/60.0, 2)-1));
    float m = v-C;
    float r,g,b;
    if(H >= 0 && H < 60){
        r = C,g = X,b = 0;
    }
    else if(H >= 60 && H < 120){
        r = X,g = C,b = 0;
    }
    else if(H >= 120 && H < 180){
        r = 0,g = C,b = X;
    }
    else if(H >= 180 && H < 240){
        r = 0,g = X,b = C;
    }
    else if(H >= 240 && H < 300){
        r = X,g = 0,b = C;
    }
    else{
        r = C,g = 0,b = X;
    }
    int R = (r+m)*255;
    int G = (g+m)*255;
    int B = (b+m)*255;

    vector<int> color{R,G,B};
    return color;
}

//* pick a color function
vector<int> pickColor(int i, int all) {
    float H = min(360.0*((i+1)/static_cast<float>(all)), 360.0);
    float S = 100;
    float V = 100;
    return HSVtoRGB(H, S, V);
}

//* Function to get affine of table_frame_wrt_camera from tf
Eigen::Affine3f get_table_frame_wrt_camera()
{
    bool tferr = true;
    int ntries = 0;
    XformUtils xformUtils;
    tf::TransformListener tfListener;
    tf::StampedTransform table_frame_wrt_cam_stf;

    Eigen::Affine3f affine_table_wrt_camera;
    while (tferr)
    {
        tferr = false;
        try
        {
            tfListener.lookupTransform("camera_rgb_optical_frame", "table_frame", ros::Time(0), table_frame_wrt_cam_stf); // was rgb
        }
        catch (tf::TransformException &exception)
        {
            ROS_WARN("%s; retrying...", exception.what());
            tferr = true;
            ros::Duration(0.5).sleep(); // sleep for half a second
            ros::spinOnce();
            ntries++;
            if (ntries > 5)
            {
                ROS_WARN("did you launch robot's table_frame_wrt_cam.launch?");
                ros::Duration(1.0).sleep();
            }
        }
    }
    ROS_INFO("tf is good for table w/rt camera");
    xformUtils.printStampedTf(table_frame_wrt_cam_stf);

    tf::Transform table_frame_wrt_cam_tf = xformUtils.get_tf_from_stamped_tf(table_frame_wrt_cam_stf);
    affine_table_wrt_camera = xformUtils.transformTFToAffine3f(table_frame_wrt_cam_tf);

    return affine_table_wrt_camera;
}

//! -------------------------------------------------------------------------------------------------------
//! Below are cloud processing functions
//! -------------------------------------------------------------------------------------------------------

//* This function will filter the point w.r.t. the z-axis; points within a certain range in z-axis are kept.
void find_indices_of_plane_from_patch(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ptr, vector<int> &indices)
{
    pcl::PassThrough<pcl::PointXYZRGB> pass;    // create a pass-through object
    pass.setInputCloud(cloud_ptr);              // set the cloud we want to operate on--pass via a pointer
    pass.setFilterFieldName("z");               // we will "filter" based on points that lie within some range of z-value
    pass.setFilterLimits(-0.01, 0.05);           // retain points with z values between these limits
    pass.filter(indices);                       // this will return the indices of the points in given cloud that pass our test
    cout << "number of points passing the filter = " << indices.size() << endl;
    //This fnc populates the reference arg "indices", so the calling fnc gets the list of interesting points
}

// This function is to segment the cloud into meaningful cluster. 
bool clusters_segmentation_cb(detect_objects::ClustersSegmentationTriggerRequest &request, detect_objects::ClustersSegmentationTriggerResponse &response)
{
    ros::NodeHandle& nh_ref = *nh_ptr;
    ros::Subscriber pointcloud_subscriber = nh_ref.subscribe("/captured_pcl", 1, cameraCB);
    got_called = true;
    cluster_cloud_ptr_vec.clear();
    //! -------------------------------------------------------------------
    
    // reset the flag to check for kinect image
    found_pcl_snapshot = false;

    // callback will run and check if there is data
    // spin until obtain a snapshot
    ROS_INFO("waiting for kinect data");
    while (!found_pcl_snapshot) {
        ROS_INFO("waiting...");
        ros::spinOnce();
        ros::Duration(0.5).sleep();
    }

    // if out of this loop -> obtained a cloud and store in input_cloud_ptr
    ROS_INFO("receive data from kinect");
        
    //! ------------------------------------------------------------------

    ROS_INFO("instantiating a pclUtils object");
    PclUtils pclUtils(nh_ptr);
    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr input_wrt_table_cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr inbound_input_wrt_table_cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr table_and_blocks_cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr blocks_cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_blocks_cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);

    pcl::toROSMsg(*input_cloud_ptr, input_pcl); //convert from PCL cloud to ROS message this way
    input_pcl.header.frame_id = "camera_rgb_optical_frame";

    //find the transform of table w/rt camera and convert to an affine
    Eigen::Affine3f affine_table_wrt_cam, affine_cam_wrt_table;
    affine_table_wrt_cam = get_table_frame_wrt_camera();
    affine_cam_wrt_table = affine_table_wrt_cam.inverse();

    //* Transform the cloud to be w.r.t. a set frame.
    //* For this case the set frame is the table_frame.
    pclUtils.transform_cloud(affine_cam_wrt_table, input_cloud_ptr, input_wrt_table_cloud_ptr);
    pcl::toROSMsg(*input_wrt_table_cloud_ptr, input_wrt_table_pcl);
    input_wrt_table_pcl.header.frame_id = "table_frame";

    // // filter outlier using statisticaloutlier removal
    // pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
    // sor.setInputCloud(input_wrt_table_cloud_ptr);
    // sor.setMeanK(100);
    // sor.setStddevMulThresh(0.7);
    // sor.filter(*filtered_input_wrt_table_cloud_ptr);
    // pcl::toROSMsg(*filtered_input_wrt_table_cloud_ptr, filtered_input_wrt_table_pcl);
    // filtered_input_wrt_table_pcl.header.frame_id = "table_frame";
    
    //! Further remove out-of-bound points
    vector<int> indices;
    Eigen::Vector3f box_pt_min, box_pt_max;
    box_pt_min << -1, -1, -0.005;
    box_pt_max << 1, 0.2, 0.05;
    pclUtils.box_filter(input_wrt_table_cloud_ptr, box_pt_min, box_pt_max, indices);
    pcl::copyPointCloud(*input_wrt_table_cloud_ptr, indices, *inbound_input_wrt_table_cloud_ptr); //extract these pts into new cloud
    pcl::toROSMsg(*inbound_input_wrt_table_cloud_ptr, inbound_input_wrt_table_pcl);           //convert to ros message for publication and display
    inbound_input_wrt_table_pcl.header.frame_id = "table_frame";

    //! remove table's points
    find_indices_of_plane_from_patch(inbound_input_wrt_table_cloud_ptr, indices);
    pcl::copyPointCloud(*inbound_input_wrt_table_cloud_ptr, indices, *table_and_blocks_cloud_ptr); //extract these pts into new cloud

    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());
    // Create the segmentation object
    pcl::SACSegmentation<pcl::PointXYZRGB> seg;
    // Optional
    seg.setOptimizeCoefficients (true);
    // Mandatory
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setDistanceThreshold (0.01);
    seg.setInputCloud (table_and_blocks_cloud_ptr);
    seg.segment (*inliers, *coefficients);

    pcl::ExtractIndices<pcl::PointXYZRGB> extract;
    extract.setInputCloud (table_and_blocks_cloud_ptr);
    extract.setIndices (inliers);
    extract.setNegative (true);
    extract.filter (*blocks_cloud_ptr);

    pcl::toROSMsg(*blocks_cloud_ptr, blocks_pcl);
    blocks_pcl.header.frame_id = "table_frame";

    //! Remove non-block points
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
    sor.setInputCloud(blocks_cloud_ptr);
    sor.setMeanK(50);
    sor.setStddevMulThresh(0.075);
    sor.filter(*filtered_blocks_cloud_ptr);
    pcl::toROSMsg(*filtered_blocks_cloud_ptr, filtered_blocks_pcl);
    filtered_blocks_pcl.header.frame_id = "table_frame";

    //! Cluster segmentation
    // Creating the KdTree object for the search method of the extraction
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
    tree->setInputCloud (filtered_blocks_cloud_ptr);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
    ec.setClusterTolerance (0.02); // 3cm
    ec.setMinClusterSize (100);
    ec.setMaxClusterSize (25000);
    ec.setSearchMethod (tree);
    ec.setInputCloud (filtered_blocks_cloud_ptr);
    ec.extract (cluster_indices);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr clusters_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    int all = cluster_indices.size();
    int j = 0;

    cout<< "clusters list" << endl;


    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
    {   
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr temp_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
        float max_z = 0;
        vector<int> color = pickColor(j, all);
        for (const auto& idx : it->indices){
            pcl::PointXYZRGB point = (*filtered_blocks_cloud_ptr)[idx];
            if (max_z < point.z) max_z = point.z;
        }
        for (const auto& idx : it->indices){
            pcl::PointXYZRGB point = (*filtered_blocks_cloud_ptr)[idx];
            point.r = color[0];
            point.g = color[1];
            point.b = color[2];
            point.z = max_z;
            clusters_cloud->push_back(point);
            temp_cloud->push_back (point);
        }
        cout<<"index: " << j << std::setw(3) << " -> " ;
        cout<< "R: " << std::setw(4) << color[0] << " || " ;
        cout<< "G: " << std::setw(4) <<  color[1] << " || " ;
        cout<< "B: " << std::setw(4) << color[2] <<endl;

        cluster_cloud_ptr_vec.push_back(temp_cloud);
        j++;
    }

    pcl::toROSMsg(*clusters_cloud, clusters_pcl);
    clusters_pcl.header.frame_id = "table_frame";

    ROS_INFO("assign num_cluster: %d", cluster_cloud_ptr_vec.size());
    response.num_clusters = j;
    if (response.num_clusters > 0)
        response.success = true;
    else
        response.success = false;
    return true;
}

bool get_cluster_cb(detect_objects::ClusterSrvMsgRequest &request, detect_objects::ClusterSrvMsgResponse &response)
{
    int size = cluster_cloud_ptr_vec.size() - 1;
    if (request.cluster_index > size)
    {
        ROS_ERROR("Cannot found any cluster with given index");
    }
    else
    {
        sensor_msgs::PointCloud2 request_pcl;
        pcl::toROSMsg(*cluster_cloud_ptr_vec[request.cluster_index], request_pcl);
        request_pcl.header.frame_id = "table_frame";
        response.cluster_pcl = request_pcl;
    }
    return true;
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "clusters_segmentation"); //node name
    ros::NodeHandle nh;
    nh_ptr = &nh;

    // create a service server that return the location of an object (if there exists one)
    ros::ServiceServer clusters_segmentation_service = nh.advertiseService("clusters_segmentation", clusters_segmentation_cb);
    ros::ServiceServer get_cluster_service = nh.advertiseService("get_cluster", get_cluster_cb);
    ros::Publisher debug_pcl = nh.advertise<sensor_msgs::PointCloud2>("debug_pcl",1);

    ROS_INFO("cluster_segmentation_service is ON");
    ROS_INFO("prepared to check for block clusters on table");
    
    while (ros::ok())
    {   
        if (got_called == true)
        {   
            debug_pcl.publish(clusters_pcl);
        }
        ros::spinOnce(); //pclUtils needs some spin cycles to invoke callbacks for new selected points
        ros::Duration(0.1).sleep();
    }

    return 0;
}

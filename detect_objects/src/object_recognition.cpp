#include <ros/ros.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>

#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h> //to convert between PCL and ROS
#include <pcl/conversions.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/surface/concave_hull.h>

#include <detect_objects/ClusterSrvMsg.h>
#include <detect_objects/ClustersSegmentationTrigger.h>

#include <geometry_msgs/PolygonStamped.h>
#include <geometry_msgs/Polygon.h>
#include <geometry_msgs/Point32.h>

using namespace std;

//! ----------------------------------------------------------------------

ros::NodeHandle* nh_ptr;
int num_clusters = 0;
sensor_msgs::PointCloud2 cluster_pcl;
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster_cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>); //pointer for color version of pointcloud
pcl::PointCloud<pcl::PointXYZRGB>::Ptr hull_cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
geometry_msgs::PolygonStamped hull_poly_stamped, simplified_hull_poly_stamped;

void clusters_segmentation()
{
    ros::ServiceClient clusters_segmentation_client = nh_ptr->serviceClient<detect_objects::ClustersSegmentationTrigger>("clusters_segmentation");
    detect_objects::ClustersSegmentationTrigger srv;
    if (clusters_segmentation_client.call(srv))
    {
        num_clusters = srv.response.num_clusters;
        ROS_INFO("Number of clusters = %d", num_clusters);
    }
    else
    {
        ROS_ERROR("Failed to call service cluster_segmentation");
    }
}

bool get_cluster()
{
    ros::ServiceClient get_cluster_client = nh_ptr->serviceClient<detect_objects::ClusterSrvMsg>("get_cluster");
    detect_objects::ClusterSrvMsg srv;
    
    cout << "Enter cluster index: ";
    int index;
    cin >> index;
    srv.request.cluster_index = index;
    
    if (get_cluster_client.call(srv))
    {
        cluster_pcl = srv.response.cluster_pcl;
        pcl::fromROSMsg(cluster_pcl, *cluster_cloud_ptr);
        return true;
    }
    else
    {
        ROS_ERROR("Failed to call service cluster_segmentation");
        return false;
    }
}

void hull_cloud_to_polygon(){
    int hull_size = hull_cloud_ptr->size();
    for (int i = 0; i < hull_size; i++)
    {
        geometry_msgs::Point32 point;
        point.x = hull_cloud_ptr->points[i].x;
        point.y = hull_cloud_ptr->points[i].y;
        point.z = hull_cloud_ptr->points[i].z;
        hull_poly_stamped.polygon.points.push_back(point);
    }
    hull_poly_stamped.header.frame_id = "table_frame";
}

float perpendicular_distance(geometry_msgs::Point32 point, geometry_msgs::Point32 linepoint1, geometry_msgs::Point32 linepoint2)
{
    float distance;
    Eigen::Vector2f vec1, vec2;
    vec1.x() = point.x - linepoint1.x;
    vec1.y() = point.y - linepoint1.y;
    vec2.x() = linepoint2.x - linepoint1.x;
    vec2.y() = linepoint2.y - linepoint1.y;

    if (vec2.norm() == 0)
        distance = vec1.norm();
    else
    {
        float dotp = vec1.dot(vec2);
        float angle = acos(dotp/(vec1.norm()*vec2.norm()));
        distance = vec1.norm()*sin(angle);
    }
    return distance;
}

// Douglas Peucker polygon simplying algorithm
std::vector<geometry_msgs::Point32> simplify_poly(std::vector<geometry_msgs::Point32> points, float epsilon){
    // std::vector<geometry_msgs::Point32> points = hull_poly_stamped.polygon.points;
    // points.push_back(points[0]);            // wrap around for Douglas Peucker algorithm
    float dmax = 0;
    int index = -1;
    int npts = points.size();

    for (int i = 1; i < npts - 1; i++)
    {
        float d = perpendicular_distance(points[i], points[0], points.back());
        if (d > dmax)
        {
            index = i;
            dmax = d;
        }
    }


    std::vector<geometry_msgs::Point32> out;

    if (dmax > epsilon)
    {
        std::vector<geometry_msgs::Point32> subVec1, subVec2;

        for (int i = 0; i <= index; i++) subVec1.push_back(points[i]);
        for (int i = index; i < points.size(); i++) subVec2.push_back(points[i]);

        std::vector<geometry_msgs::Point32> recResult1 = simplify_poly(subVec1, epsilon);
        std::vector<geometry_msgs::Point32> recResult2 = simplify_poly(subVec2, epsilon);

        for (int i = 0; i < recResult1.size() - 1; i++) out.push_back(recResult1[i]);
        for (int i = 0; i < recResult2.size(); i++) out.push_back(recResult2[i]);
    }
    else
    {
        out.push_back(points[0]);
        out.push_back(points.back());
    }
    return out;
}

int main (int argc, char **argv)
{
    ros::init(argc, argv, "object_recognition");
    ros::NodeHandle nh;
    nh_ptr = &nh;
    ros::Publisher hull_cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("hull_pcl",1);
    ros::Publisher hull_poly_pub = nh.advertise<geometry_msgs::PolygonStamped>("hull_poly",1);
    ros::Publisher simplified_hull_poly_pub = nh.advertise<geometry_msgs::PolygonStamped>("simplified_hull_poly",1);


    clusters_segmentation();
    get_cluster();
    
    if (cluster_cloud_ptr->size() == 0)
    {
        ROS_ERROR("Empty cluster!!! Check available detected clusters.");\
        return 0;
    }

    // Create a Convex Hull representation of the projected inliers
    pcl::ConvexHull<pcl::PointXYZRGB> chull;
    chull.setInputCloud (cluster_cloud_ptr);
    chull.reconstruct (*hull_cloud_ptr);

    cout << "Convex hull has: " << hull_cloud_ptr->size () << " data points." << std::endl;

    hull_cloud_to_polygon();

    int num_vertices = INT_MAX;

    std::vector<geometry_msgs::Point32> points = hull_poly_stamped.polygon.points;
    for (int i = 0; i < hull_poly_stamped.polygon.points.size(); i++)
    {
        // circular shifting left
        std::rotate(points.begin(),points.begin()+i,points.end());
        // wrap around for Douglas Peucker
        points.push_back(points[0]);
        // Dougals Peucker
        std::vector<geometry_msgs::Point32> simplified_poly = simplify_poly(points, 0.005);
        simplified_poly.pop_back();
        if (simplified_poly.size() < num_vertices)
        {
            num_vertices = simplified_poly.size();              //! find minimum number of vertices
            simplified_hull_poly_stamped.header.frame_id = hull_poly_stamped.header.frame_id;
            simplified_hull_poly_stamped.polygon.points = simplified_poly;
        }
    }

    ROS_INFO("Number of vertices: %d", num_vertices);

    while (ros::ok())
    {   
        hull_cloud_pub.publish(hull_cloud_ptr);
        hull_poly_pub.publish(hull_poly_stamped);
        simplified_hull_poly_pub.publish(simplified_hull_poly_stamped);

        ros::spinOnce(); //pclUtils needs some spin cycles to invoke callbacks for new selected points
        ros::Duration(0.1).sleep();
    }

    return 0;
}
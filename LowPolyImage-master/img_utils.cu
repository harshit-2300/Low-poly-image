#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "point.h"
#include "triangle.h"

using namespace std;

// Draw the edge detection images
cv::Mat drawEdges(uint8_t* gradient_img, int height, int width)
{
    cv::Mat edge_output = cv::Mat(height, width, CV_8UC1, cv::Scalar(0));
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            edge_output.at<uchar>(i, j) = gradient_img[i * width + j];
        }
    }
    return edge_output;
}


// Draw the selected vertex images
cv::Mat drawVert(Point *vert_img, int height, int width)
{
    int total_pixels = height * width;
    cv::Mat vertex_output = cv::Mat(height, width, CV_8UC1, cv::Scalar(0));
    for (int i = 0; i < total_pixels; ++i)
    {
        Point p = vert_img[i];
        if (!p.isInvalid())
        {
            int x = p.x;
            int y = p.y;
            vertex_output.at<uchar>(y, x) = 255;
        }
    }
    return vertex_output;
}


// Draw the voroni images
cv::Mat drawVoroni(Point *owner, int height, int width)
{
    cv::Mat voroni_output = cv::Mat(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    
    // Randomized RGB color for each region
    int total_pixels = height * width;
    vector<cv::Vec3b> vertices_color(total_pixels);
    for (int i = 0; i < total_pixels; i++)
    {
        vertices_color[i][0] = rand() % 256;
        vertices_color[i][1] = rand() % 256;
        vertices_color[i][2] = rand() % 256;
    }

    // Assign each pixel with corresponding region color
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            Point p = owner[i * width + j];
            voroni_output.at<cv::Vec3b>(i, j) = vertices_color[p.y * width + p.x];
        }
    }

    return voroni_output;
}


// Draw the triangulation results
cv::Mat drawTriangles(vector<Triangle> &triangles, cv::Mat &img, bool add)
{
    cv::Mat triangle_output;
    // there are two different operations
    // add is True : overwrite the input image with triangulation results
    // add is False: only write triangulation results in a new image
    if (add)
    {
        triangle_output = img.clone();
        for (int i = 0; i < triangles.size(); i++)
        {
            Triangle tri = triangles[i];
            cv::Point p1, p2, p3;
            p1.x = tri.points[0].x;
            p1.y = tri.points[0].y;
            p2.x = tri.points[1].x;
            p2.y = tri.points[1].y;
            p3.x = tri.points[2].x;
            p3.y = tri.points[2].y;
            cv::line(triangle_output, p1, p2, cv::Scalar( 0, 0, 0));
            cv::line(triangle_output, p2, p3, cv::Scalar( 0, 0, 0));
            cv::line(triangle_output, p3, p1, cv::Scalar( 0, 0, 0));
        }
    }
    else
    {
        int height = img.rows;  int width = img.cols;
        triangle_output = cv::Mat(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
        for (int i = 0; i < triangles.size(); i++)
        {
            Triangle tri = triangles[i];
            cv::Point p1, p2, p3;
            p1.x = tri.points[0].x;
            p1.y = tri.points[0].y;
            p2.x = tri.points[1].x;
            p2.y = tri.points[1].y;
            p3.x = tri.points[2].x;
            p3.y = tri.points[2].y;
            cv::line(triangle_output, p1, p2, cv::Scalar( 255, 255, 255));
            cv::line(triangle_output, p2, p3, cv::Scalar( 255, 255, 255));
            cv::line(triangle_output, p3, p1, cv::Scalar( 255, 255, 255));
        }
    }
    return triangle_output;
}
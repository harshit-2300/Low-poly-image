// Avoid repeated macro defined
#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "point.h"
#include "triangle.h"

using namespace std;

// Image drawing unit
// cv::Mat drawLowPoly(vector<Triangle> &triangles, cv::Mat &orig_img, int height, int width);
cv::Mat drawTriangles(vector<Triangle> &triangles, cv::Mat &img, bool add);
cv::Mat drawEdges(uint8_t* gradient_img, int height, int width);
cv::Mat drawVert(Point *vert_img, int height, int width);
cv::Mat drawVoroni(Point *owner, int height, int width);
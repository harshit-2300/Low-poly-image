// Avoid repeated macro defined
#pragma once

#include "point.h"

#define CUDA_ENV __host__ __device__

struct Triangle
{
    Point points[3];

    CUDA_ENV
    Triangle(){}

    CUDA_ENV
    Triangle(Point a, Point b, Point c);

    CUDA_ENV
    Point get_center();
};
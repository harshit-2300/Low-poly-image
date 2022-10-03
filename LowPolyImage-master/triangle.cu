#include "triangle.h"

#define CUDA_ENV __host__ __device__

CUDA_ENV
Triangle::Triangle(Point a, Point b, Point c)
{
    points[0] = a;
    points[1] = b;
    points[2] = c;
}

CUDA_ENV
Point Triangle::get_center()
{
    return (points[0] + points[1] + points[2]) / 3;
}
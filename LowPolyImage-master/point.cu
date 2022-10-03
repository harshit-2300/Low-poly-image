#include "point.h"

#define CUDA_ENV __host__ __device__

CUDA_ENV
Point::Point(int x_, int y_)
{
    x = x_;
    y = y_;
}

CUDA_ENV
bool Point::isInvalid()
{
    return x == -1 && y == -1;
}

CUDA_ENV
Point operator + (const Point &a, const Point &b)
{
    return Point(a.x + b.x, a.y + b.y);
}

CUDA_ENV
Point operator * (const Point &a, int b)
{
    return Point(a.x * b, a.y * b);
}

CUDA_ENV
Point operator / (const Point &a, int b)
{
    return Point(a.x / b, a.y / b);
}

CUDA_ENV
bool operator == (const Point &a, const Point &b)
{
    return a.x == b.x && a.y == b.y;
}

CUDA_ENV
int distance(const Point &a, const Point &b) 
{
    int dx = a.x - b.x;
    int dy = a.y - b.y;
    return dx * dx + dy * dy;
}
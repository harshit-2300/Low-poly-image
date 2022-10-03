// Avoid repeated macro defined
#pragma once

#define CUDA_ENV __host__ __device__

struct Point
{
    int x, y;

    CUDA_ENV
    Point(){}

    CUDA_ENV
    Point(int x_, int y_);

    CUDA_ENV
    bool isInvalid();
};

CUDA_ENV
Point operator + (const Point &a, const Point &b);

CUDA_ENV
Point operator * (const Point &a, int b);

CUDA_ENV
Point operator / (const Point &a, int b);

CUDA_ENV
bool operator == (const Point &a, const Point &b);

CUDA_ENV
int distance(const Point &a, const Point &b);


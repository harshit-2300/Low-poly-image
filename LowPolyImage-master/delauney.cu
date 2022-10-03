#include <vector>
#include <unordered_set>
#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <curand_kernel.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "point.h"
#include "triangle.h"
#include "delauney.h"
#include "simpleTimer.h"

using namespace std;

#define SEED 1234

#define MASK_N 2
#define MASK_X 3
#define MASK_Y 3
#define SCALE 8

__constant__ int adjustX = (MASK_X % 2) ? 1 : 0;
__constant__ int adjustY = (MASK_Y % 2) ? 1 : 0;
__constant__ int xBound = MASK_X / 2;
__constant__ int yBound = MASK_Y / 2;

uint8_t *grey_img_GPU;
uint8_t *gradient_img_GPU;
Point *owner_map_GPU;
curandState *rand_state;
int *triangle_count_GPU;
int *triangle_prefix_sum_GPU;
Triangle *triangles_GPU;
uint8_t *orig_img_GPU;
uint8_t *color_img_GPU;

int total_triangles;


#define checkCuda(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void init_cuda()
{
    cudaFree(0);
}

__global__
void get_gradient_kernel(uint8_t *grey_img, uint8_t *gradient_img, int height, int width)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    int mask[MASK_N][MASK_X][MASK_Y] = {
        {{1, 0, -1},
         {2, 0, -2},
         {1, 0, -1}},
        {{1, 2, 1},
         {0, 0, 0},
         {-1, -2, -1}}
    };

    if (y >= 0 && y < height && x >= 0 && x < width) {
        float grad[2] = {0.0, 0.0};

        for (int i = 0; i < MASK_N; ++i) {
            grad[i] = 0.0;
            for (int v = -yBound; v < yBound + adjustY; ++v) {
                for (int u = -xBound; u < xBound + adjustX; ++u) {
                    if ((x + u) >= 0 && (x + u) < width && (y + v) >= 0 && (y + v) < height) {
                        grad[i] += grey_img[width * (y + v) + (x + u)] * mask[i][u + xBound][v + yBound];
                    }
                }
            }
            grad[i] = abs(grad[i]);
        }

        float total_grad = grad[0] / 2.0 + grad[1] / 2.0;
        const unsigned char c = (total_grad > 255.0) ? 255 : total_grad;
        gradient_img[y * width + x] = c;
    }
}

__global__
void setup_kernel(curandState *state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    curand_init( (SEED << 20) + id, 0, 0, &state[id]);

}



__global__
void select_vertices_kernel(uint8_t *grad, Point *owner_map, curandState *rand_state, int height, int width, float gradThreshold, float edgeProb, float nonEdgeProb, float boundProb)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int y = id / width;
    int x = id % width;

    uint8_t gradVal;

    if (y < 0 || y >= height || x < 0 || x >= width)
        return;

    if ((y == 0 || y == height-1) && (x == 0 || x == width-1))
    {
        Point p;
        p.x = x;
        p.y = y;
        owner_map[id] = p;
    }
    else
    {
        curandState localState = rand_state[id];
        float randNum = curand_uniform(&localState);

        // inner area
        if (y > 0 && y < height-1 && x > 0 && x < width-1)
        {
            gradVal = grad[y * width + x];
            if (gradVal > gradThreshold && randNum < edgeProb)
            {
                // Edge vertex
                Point p;
                p.x = x;
                p.y = y;
                owner_map[id] = p;
            }
            else if (randNum < nonEdgeProb)
            {
                // Non-edge vertex
                Point p;
                p.x = x;
                p.y = y;
                owner_map[id] = p;
            }
        }
        // boundary area
        else
        {
            if (randNum < boundProb)
            {
                Point p;
                p.x = x;
                p.y = y;
                owner_map[id] = p;
            }
        }
    }
}


void select_vertices_GPU(uint8_t *grey_img_CPU, uint8_t *result_image, Point *vert_img, int height, int width)
{
    simpleTimer t_edge_detect("...Edge detection");

    int total_pixels = height * width;

    // Data transfer
    checkCuda( cudaMemcpy(grey_img_GPU, grey_img_CPU, total_pixels * sizeof(uint8_t), cudaMemcpyHostToDevice) );
    // Init for owner
    checkCuda( cudaMemset(owner_map_GPU, -1, total_pixels * sizeof(Point)) );

    // Edge detection filtering
    int BLOCKSIZE = 32;
    dim3 dimBlock(BLOCKSIZE, BLOCKSIZE, 1);
    dim3 dimGrid(ceil(width/(float)BLOCKSIZE), ceil(height/(float)BLOCKSIZE), 1);
    get_gradient_kernel<<<dimGrid, dimBlock>>>(grey_img_GPU, gradient_img_GPU, height, width);
    checkCuda( cudaDeviceSynchronize() );

    t_edge_detect.GetDuration();

    simpleTimer t_init_random("...random generators init");

    // Init random generators
    int GRIDSIZE = (total_pixels + BLOCKSIZE - 1) / BLOCKSIZE;
    setup_kernel<<<GRIDSIZE, BLOCKSIZE>>>(rand_state);
    checkCuda( cudaDeviceSynchronize() );

    t_init_random.GetDuration();

    simpleTimer t_select_vert("...Vertices selection");

    // Selecting vertices
    float gradThreshold = 20;
    float edgeProb = 0.005;
    float nonEdgeProb = 0.0001;
    float boundProb = 0.1;
    select_vertices_kernel<<<GRIDSIZE, BLOCKSIZE>>>(gradient_img_GPU, owner_map_GPU, rand_state, 
                            height, width, gradThreshold, edgeProb, nonEdgeProb, boundProb);
    checkCuda( cudaDeviceSynchronize() );

    t_select_vert.GetDuration();

}


int ceil_power2(int v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

__device__
inline int convert_idx(Point p, int width)
{
    return p.y * width + p.x;
}

__device__
inline bool out_of_bound(Point p, int height, int width)
{
    return !(p.x >= 0 && p.x < width && p.y >= 0 && p.y < height);
}


__global__
void jump_flooding_kernel(Point *owner_map, int step_size, int height, int width)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    // All 8 directions to check from the vertex
    const Point all_dir[] = {Point(1, 0), Point(1, 1), Point(0, 1), Point(-1, 1),
        Point(-1, 0), Point(-1, -1), Point(0, -1), Point(1, -1)};

    if (y >= 0 && y < height && x >= 0 && x < width)
    {
        Point cur_point;
        cur_point.x = x;
        cur_point.y = y;
        // Check for all possible directions to neighbor points
        for (int i = 0; i < 8; i++)
        {
            Point cur_dir = all_dir[i];
            Point cur_looking = cur_point + cur_dir * step_size;
            // If this point is out of bounds, skip it
            if (out_of_bound(cur_looking, height, width))
            {
                continue;
            }
            // If this point is not owned by anyone, skip it
            if (owner_map[convert_idx(cur_looking, width)].isInvalid())
            {
                continue;
            }

            // Update owner in cur_point only when
            // 1. cur_point is NOT owned by anyone (owner = NULL)
            // 2. cur_point has shorter distance to cur_looking's owner than previous owner
            Point cur_owner = owner_map[convert_idx(cur_point, width)];
            int tmp_dist = distance(owner_map[convert_idx(cur_looking, width)], cur_point);
            if (cur_owner.isInvalid() || tmp_dist < distance(cur_owner, cur_point))
            {
                owner_map[convert_idx(cur_point, width)] = owner_map[convert_idx(cur_looking, width)];
            }
        }
    }
}


__device__
void check_triangles(Point *owner_map, int *num_triangles, Point cur_point, Point test[], int width)
{
    Point corner_dir[3] = {Point(0, 1), Point(1, 0), Point(1, 1)};
    
    // setup for first point
    int num_colors = 1;
    test[0] = owner_map[convert_idx(cur_point, width)];

    // test all 4 points
    for (int i = 0; i < 3; i++)
    {
        Point neighbor_point = cur_point + corner_dir[i];
        int is_diff = true;
        // check the buffer for distinct points
        for (int j = 0; j < num_colors; j++)
        {
            // same point can be found in buffer
            if (owner_map[convert_idx(neighbor_point, width)] == test[j])
            {
                is_diff = false;
                break;
            }
        }
        if (is_diff)
        {
            test[num_colors] = owner_map[convert_idx(neighbor_point, width)];
            num_colors++;
        }
    }

    if (num_colors == 3) {
        *num_triangles = 1;
    } else if (num_colors == 4) {
        *num_triangles = 2;
    } else {
        *num_triangles = 0;
    }
}


__global__
void triangle_count_kernel(int *count, Point *owner_map, int height, int width)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (y >= 0 && y < height-1 && x >= 0 && x < width-1)
    {
        int num_triangles = 0;
        Point cur_point(x, y);
        Point test[4] = {Point(-1,-1), Point(-1,-1), Point(-1,-1), Point(-1,-1)};
        check_triangles(owner_map, &num_triangles, cur_point, test, width);
        count[y * width + x] = num_triangles;
    } else {
        count[y * width + x] = 0;
    }
}


__global__
void triangle_generate_kernel(Triangle *triangles, Point *owner_map, int *prefix_sum, int height, int width)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (y >= 0 && y < height-1 && x >= 0 && x < width-1)
    {
        int num_triangles = 0;
        Point cur_point(x, y);
        Point test[4] = {Point(-1,-1), Point(-1,-1), Point(-1,-1), Point(-1,-1)};
        check_triangles(owner_map, &num_triangles, cur_point, test, width);

        if (num_triangles == 1)
        {
            Triangle triangle;
            triangle = Triangle(owner_map[convert_idx(test[0], width)],
                                owner_map[convert_idx(test[1], width)],
                                owner_map[convert_idx(test[2], width)]);
            int triangle_index = prefix_sum[y * width + x] - 1;
            triangles[triangle_index] = triangle;
        }
        else if (num_triangles == 2)
        {
            Triangle triangle1, triangle2;
            triangle1 = Triangle(owner_map[convert_idx(test[0], width)],
                                owner_map[convert_idx(test[1], width)],
                                owner_map[convert_idx(test[2], width)]);
            triangle2 = Triangle(owner_map[convert_idx(test[1], width)],
                                owner_map[convert_idx(test[2], width)],
                                owner_map[convert_idx(test[3], width)]);
            int triangle_index = prefix_sum[y * width + x] - 2;
            triangles[triangle_index] = triangle1;
            triangles[triangle_index + 1] = triangle2;
        }
    }
}


void delauney_GPU(Point *owner_map_CPU, vector<Triangle> &triangles_list, int height, int width)
{
    int total_pixels = height * width;
    int BLOCKSIZE = 32;
    dim3 dimBlock(BLOCKSIZE, BLOCKSIZE, 1);
    dim3 dimGrid(ceil(width/(float)BLOCKSIZE), ceil(height/(float)BLOCKSIZE), 1);

    simpleTimer t_jump_flood("...Jump flooding");

    int init_step_size = ceil_power2(min(height, width)) / 2;
    // Iterate possible step sizes
    for (int step_size = init_step_size; step_size >= 1; step_size /= 2)
    {
        jump_flooding_kernel<<<dimGrid, dimBlock>>>(owner_map_GPU, step_size, height, width);
        checkCuda( cudaDeviceSynchronize() );
    }

    t_jump_flood.GetDuration();

    // **************************************
    // Building triangles from the voronoi diagram
    // (1) get triangle count for each pixel
    // (2) compute prefix sum of the number of triangles
    // (3) perform triangulation
    // **************************************

    simpleTimer t_build_tri("...Building triangles");

    simpleTimer t_count("......Counting triangles");

    // **************************************
    // (1) get triangle count for each pixel
    // **************************************
    checkCuda( cudaMalloc(&triangle_count_GPU, total_pixels * sizeof(int)) );
    triangle_count_kernel<<<dimGrid, dimBlock>>>(triangle_count_GPU, owner_map_GPU, height, width);
    checkCuda( cudaDeviceSynchronize() );

    t_count.GetDuration();

    simpleTimer t_prefix("......Counting prefix sum");

    // **************************************
    // (2) compute prefix sum of the number of triangles
    // **************************************
    checkCuda( cudaMalloc(&triangle_prefix_sum_GPU, total_pixels * sizeof(int)) );
    thrust::inclusive_scan(thrust::device, triangle_count_GPU, triangle_count_GPU + total_pixels, triangle_prefix_sum_GPU);
    checkCuda( cudaMemcpy(&total_triangles, &triangle_prefix_sum_GPU[total_pixels-1], sizeof(int), cudaMemcpyDeviceToHost) );
    
    t_prefix.GetDuration();

    simpleTimer t_tri("......Triangulation");

    // **************************************
    // (3) perform triangulation
    // **************************************
    checkCuda( cudaMalloc(&triangles_GPU, total_triangles * sizeof(Triangle)) );
    triangle_generate_kernel<<<dimGrid, dimBlock>>>(triangles_GPU, owner_map_GPU, triangle_prefix_sum_GPU, height, width);
    checkCuda( cudaDeviceSynchronize() );

    t_tri.GetDuration();

    t_build_tri.GetDuration();

    std::cout << "total number of triangles processed: " << total_triangles << std::endl;

}

__device__ float sign(Point p1, Point p2, Point p3)
{
    return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y);
}

__device__ bool PointInTriangle(Point pt, Point v1, Point v2, Point v3)
{
    float d1, d2, d3;
    bool has_neg, has_pos;

    d1 = sign(pt, v1, v2);
    d2 = sign(pt, v2, v3);
    d3 = sign(pt, v3, v1);

    has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
    has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);

    return !(has_neg && has_pos);
}


__global__
void draw_lowpoly_kernel(uint8_t *orig_img, uint8_t *color_img, Triangle *triangles, int total_triangles, int width)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > total_triangles)
        return;

    Triangle triangle = triangles[idx];
    // Use center pixel of a triangle to color it
    Point pt_c = triangle.get_center();
    // Find bounding box region of a triangle
    int minX = min(triangle.points[0].x, min(triangle.points[1].x, triangle.points[2].x));
    int maxX = max(triangle.points[0].x, max(triangle.points[1].x, triangle.points[2].x));
    int minY = min(triangle.points[0].y, min(triangle.points[1].y, triangle.points[2].y));
    int maxY = max(triangle.points[0].y, max(triangle.points[1].y, triangle.points[2].y));

    int img_idx = (pt_c.y * width + pt_c.x) * 3;
    // Iterate for the pixels in the box region
    for (int y = minY; y <= maxY; y++)
    {
        for (int x = minX; x <= maxX; x++)
        {
            Point pt_tmp(x, y);
            // Check if the pixels lies in the triangle
            if (PointInTriangle(pt_tmp, triangle.points[0], triangle.points[1], triangle.points[2]))
            {
                int tri_img_idx = (y * width + x) * 3;
                // Assign the color of ceter pixel of the triangle to current pixel
                color_img[tri_img_idx] = orig_img[img_idx];
                color_img[tri_img_idx + 1] = orig_img[img_idx + 1];
                color_img[tri_img_idx + 2] = orig_img[img_idx + 2];
            }
        }
    }
}

__global__
void draw_lowpoly_stride_kernel(uint8_t *orig_img, uint8_t *color_img, Triangle *triangles, int total_triangles, int width)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > total_triangles)
        return;

    for (int i = idx; i < total_triangles; i += gridDim.x * blockDim.x)
    {
        Triangle triangle = triangles[i];
        // Use center pixel of a triangle to color it
        Point pt_c = triangle.get_center();
        // Find bounding box region of a triangle
        int minX = min(triangle.points[0].x, min(triangle.points[1].x, triangle.points[2].x));
        int maxX = max(triangle.points[0].x, max(triangle.points[1].x, triangle.points[2].x));
        int minY = min(triangle.points[0].y, min(triangle.points[1].y, triangle.points[2].y));
        int maxY = max(triangle.points[0].y, max(triangle.points[1].y, triangle.points[2].y));

        int img_idx = (pt_c.y * width + pt_c.x) * 3;
        // Iterate for the pixels in the box region
        for (int y = minY; y <= maxY; y++)
        {
            for (int x = minX; x <= maxX; x++)
            {
                Point pt_tmp(x, y);
                // Check if the pixels lies in the triangle
                if (PointInTriangle(pt_tmp, triangle.points[0], triangle.points[1], triangle.points[2]))
                {
                    int tri_img_idx = (y * width + x) * 3;
                    // Assign the color of ceter pixel of the triangle to current pixel
                    color_img[tri_img_idx] = orig_img[img_idx];
                    color_img[tri_img_idx + 1] = orig_img[img_idx + 1];
                    color_img[tri_img_idx + 2] = orig_img[img_idx + 2];
                }
            }
        }
    }
}


__global__
void compute_area_kernel(int *triangle_area, Triangle *triangles, int total_triangles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > total_triangles)
        return;

    Triangle triangle = triangles[idx];
    // Find bounding box region of a triangle
    int minX = min(triangle.points[0].x, min(triangle.points[1].x, triangle.points[2].x));
    int maxX = max(triangle.points[0].x, max(triangle.points[1].x, triangle.points[2].x));
    int minY = min(triangle.points[0].y, min(triangle.points[1].y, triangle.points[2].y));
    int maxY = max(triangle.points[0].y, max(triangle.points[1].y, triangle.points[2].y));
    // Get the length of two dimensions
    int x_length = ((maxX - minX) < 1) ? 1 : maxX - minX;
    int y_length = ((maxY - minY) < 1) ? 1 : maxY - minY;

    triangle_area[idx] = x_length * y_length;
}


cv::Mat drawLowPoly_GPU(cv::Mat &img)
{
    int height = img.rows;
    int width = img.cols;
    int total_pixels = height * width;

    checkCuda( cudaMemcpy(orig_img_GPU, img.data, sizeof(uint8_t) * total_pixels * 3, cudaMemcpyHostToDevice) );

    int BLOCKSIZE = 64;
    int GRIDSIZE = (total_triangles + BLOCKSIZE - 1) / BLOCKSIZE;

    // ********************************
    // Compute each triangle area
    int *triangle_area_GPU;
    cudaMalloc(&triangle_area_GPU, sizeof(int) * total_triangles);
    compute_area_kernel<<<GRIDSIZE, BLOCKSIZE>>>(triangle_area_GPU, triangles_GPU, total_triangles);
    checkCuda( cudaDeviceSynchronize() );
    // Sort the triangle first
    thrust::sort_by_key(thrust::device, triangle_area_GPU, triangle_area_GPU + total_triangles, triangles_GPU);
    cudaFree(triangle_area_GPU);
    // ********************************

    // Normal Version
    draw_lowpoly_kernel<<<GRIDSIZE, BLOCKSIZE>>>(orig_img_GPU, color_img_GPU, triangles_GPU, total_triangles, width);

    // Strided Version
    // draw_lowpoly_stride_kernel<<<20 * 32, 64>>>(orig_img_GPU, color_img_GPU, triangles_GPU, total_triangles, width);

    checkCuda( cudaDeviceSynchronize() );

    cv::Mat color_img;
    color_img.create(height, width, CV_8UC3);
    cudaMemcpy(color_img.data, color_img_GPU, sizeof(uint8_t) * total_pixels * 3, cudaMemcpyDeviceToHost);

    return color_img;
}


void setup_gpu_memory(int height, int width)
{
    int total_pixels = height * width;

    // GPU memory allocation
    checkCuda( cudaMalloc(&grey_img_GPU, total_pixels * sizeof(uint8_t)) );
    checkCuda( cudaMalloc(&gradient_img_GPU, total_pixels * sizeof(uint8_t)) );
    checkCuda( cudaMalloc(&owner_map_GPU, total_pixels * sizeof(Point)) );
    checkCuda( cudaMalloc(&rand_state, total_pixels * sizeof(curandState)) );
    checkCuda( cudaMalloc(&orig_img_GPU, sizeof(uint8_t) * total_pixels * 3) );
    checkCuda( cudaMalloc(&color_img_GPU, sizeof(uint8_t) * total_pixels * 3) );
}


void free_gpu_memory()
{
    checkCuda( cudaFree(grey_img_GPU) );
    checkCuda( cudaFree(gradient_img_GPU) );
    checkCuda( cudaFree(owner_map_GPU) );
    checkCuda( cudaFree(rand_state) );
    checkCuda( cudaFree(triangle_count_GPU) );
    checkCuda( cudaFree(triangle_prefix_sum_GPU) );
    checkCuda( cudaFree(triangles_GPU) );
    checkCuda( cudaFree(orig_img_GPU) );
    checkCuda( cudaFree(color_img_GPU) );
}
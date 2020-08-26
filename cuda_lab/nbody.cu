#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include "check.h"

#define SOFTENING 1e-9f

typedef struct {
    float x, y, z;
    float vx, vy, vz;
} Body;

typedef enum {X, Y, Z} Axis;

void randomizeBodies(float *data, int n)
{
    for (int i = 0; i < n; i++) {
        data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    }
}

__global__ void bodyForce(Body *p, float dt, int n)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    float Fx, Fy, Fz;
    float dx, dy, dz;
    float distSqr, invDist, invDist3;

    for (int i = tid; i < n; i += stride) {
        Fx = 0.0f;
        Fy = 0.0f;
        Fz = 0.0f;

        for (int j = 0; j < n; j++) {
            dx = p[j].x - p[i].x;
            dy = p[j].y - p[i].y;
            dz = p[j].z - p[i].z;

            distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
            invDist = rsqrtf(distSqr);
            invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        p[i].vx += dt * Fx;
        p[i].vy += dt * Fy;
        p[i].vz += dt * Fz;
    }
}

__global__ void integratePos(Body *p, float dt, int n, int axis)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    switch (axis) {
        case X:
            for (int i = tid; i < n; i += stride) {
                p[i].x += p[i].vx * dt;
            }
            break;
        case Y:
            for (int i = tid; i < n; i += stride) {
                p[i].y += p[i].vy * dt;
            }
            break;
        case Z:
            for (int i = tid; i < n; i += stride) {
                p[i].z += p[i].vz * dt;
            }
            break;
        default:
            printf("Unknown axis in function `integratePos`.\n");
    }
}

int main(const int argc, const char** argv)
{
    int nBodies = 2 << 11;
    if (argc > 1) nBodies = 2 << atoi(argv[1]);

    int salt = 0;
    if (argc > 2) salt = atoi(argv[2]);

    const float dt = 0.01f; // time step
    const int nIters = 10;  // simulation iterations

    int bytes = nBodies * sizeof(Body);
    float *buf;

    cudaMallocManaged(&buf, bytes);
    Body *p = (Body*)buf;

    int device_id;
    cudaGetDevice(&device_id);
    
    randomizeBodies(buf, 6 * nBodies); // Init pos / vel data

    cudaMemPrefetchAsync(p, bytes, device_id);
    
    int num_blocks, num_threads;
    
    cudaDeviceGetAttribute(&num_blocks, cudaDevAttrMultiProcessorCount, device_id);
    num_threads = 64;

    /*******************************************************************/
    double totalTime = 0.0;
    cudaStream_t stream;

    for (int iter = 0; iter < nIters; iter++) {
        StartTimer();
        /*******************************************************************/

        bodyForce<<<num_blocks, num_threads>>>(p, dt, nBodies); // compute interbody forces

        for (int ax = 0; ax < 3; ++ax) {
            cudaStreamCreate(&stream);
            integratePos<<<num_blocks, num_threads, 0, stream>>>(p, dt, nBodies, ax); // integrate position
            cudaStreamDestroy(stream);
        }

        /*******************************************************************/
        // Do not modify the code in this section.
        const double tElapsed = GetTimer() / 1000.0;
        totalTime += tElapsed;
    }
    cudaDeviceSynchronize();

    double avgTime = totalTime / (double)(nIters);
    float billionsOfOpsPerSecond = 1e-9 * nBodies * nBodies / avgTime;

    #ifdef ASSESS
    checkPerformance(buf, billionsOfOpsPerSecond, salt);
    #else
    checkAccuracy(buf, nBodies);
    printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies, billionsOfOpsPerSecond);
    salt += 1;
    #endif
    /*******************************************************************/

    cudaFree(buf);
}

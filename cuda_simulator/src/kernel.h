#pragma once
#include <assert.h>
#include <cuda_runtime.h>
#include <particle_io.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "log.h"

#define BUCKET_CAPACITY 16
#define BUCKETS_X 32
#define BUCKETS_Y 32
#define BUCKETS_COUNT (BUCKETS_X * BUCKETS_Y)
#define MAX_PARTICLE_COUNT (BUCKET_CAPACITY * BUCKETS_COUNT)

static Particle* k_0;
static Particle* k_1;
static Particle* k_internal;
static FrameHeader* frame;

bool running_with_gpus;

// # Desmos
// C\ =\ \frac{n}{n-m}\left(\frac{n}{m}\right)^{\frac{m}{n-m}}
// F\left(r\right)=C\cdot p\cdot\frac{m\left(\frac{s}{r}\right)^{m}-n\left(\frac{s}{r}\right)^{n}}{r}
// V\left(r\right)=C\cdot p\left(\left(\frac{s}{r}\right)^{n}-\left(\frac{s}{r}\right)^{m}\right)
//
// # Wolframalpha
// Partial[\(40)Power[\(40)Divide[s,x]\(41),n]-Power[\(40)Divide[s,x]\(41),m]\(41) ,x]

__global__ static void gpu_kernel(Particle* src, Particle* dst, FrameMetadata frame, uint32_t count) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    Particle p = src[i];
    
    float fx = 0;
    float fy = 0;
    
    float k_b = 1.380649e-23;
    float mass = 6.63352599e-26;
    MiePotentialParams params = frame.particles[0];
    float C = (params.n / (params.n - params.m)) * powf(params.n / params.m, params.n / (params.n - params.m));
    params.epsilon *= k_b;
 
    for (uint32_t j = 0; j < count; ++j) {
        if (j == i) continue;

        float rx = src[j].x - src[i].x;
        float ry = src[j].y - src[i].y;
        float r = hypotf(rx, ry);
        float sr = params.sigma / r;

        float f = C * params.epsilon * (params.m * powf(sr, params.m) - params.n * powf(sr, params.n)) / r;
        // Alternative:  __fsqrt_rn(float)
        
        // Normalize vector r
        rx /= r;
        ry /= r;

        fx += f * rx;
        fy += f * ry;
    }    

    // f = m v/dt
    // f dt / m = v
    dst[i].vx = p.vx + fx * frame.step_dt / mass;
    dst[i].vy = p.vy + fy * frame.step_dt / mass;

    dst[i].x = p.x + p.vx * frame.step_dt;
    dst[i].y = p.y + p.vy * frame.step_dt;

    dst[i].ty = p.ty;

    if (dst[i].x < 0. || dst[i].x >= frame.box_width)
        dst[i].vx = -dst[i].vx;
    if (dst[i].y < 0. || dst[i].y >= frame.box_height)
        dst[i].vy = -dst[i].vy;
}

static void cpu_kernel(Particle* src, Particle* dst, float dt, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        Particle p = src[i];

        dst[i].x = p.x + p.vx * dt;
        dst[i].y = p.y + p.vy * dt;
        dst[i].vx = p.vx;
        dst[i].vy = p.vy;
        dst[i].ty = p.ty;
    }
}

static void step(Particle* src, Particle* dst, FrameMetadata frame, uint32_t count) {
    // cpu_kernel(src, dst, dt, count);

    uint32_t nThreads = 256;
    uint32_t nBlocks = (count + nThreads - 1) / nThreads;
    gpu_kernel<<<nBlocks, nThreads>>>(src, dst, frame, count);
}

static void run_kernel_async(FrameHeader* frame, Particle* k_src, Particle* k_dst) {
    uint32_t count = frame->particles_count;

    if (frame->metadata.steps_per_frame & 1 == 0) {
        step(k_src, k_internal, frame->metadata, count);
        step(k_internal, k_dst, frame->metadata, count);
    } else {
        step(k_src, k_dst, frame->metadata, count);
    }

    for (uint32_t i = 2; i < frame->metadata.steps_per_frame; i += 2) {
        step(k_dst, k_internal, frame->metadata, count);
        step(k_internal, k_dst, frame->metadata, count);
    }
}

static void sync_kernel() {
    // cudaSync(stream);
}

// Convert a list of particles into the data-structure used by the kernel
static void frame_prepare(FrameHeader* src, FrameHeader* dst) {
    memcpy((void*)dst, (void*)src, packet_size(src->particles_count));

    // Compact array
    // frame_compact_into(src, dst);

    // // Bucket Matrix
    // uint32_t bucket_len[BUCKETS_X * BUCKETS_Y];
    // memset(bucket_len, 0, sizeof(bucket_len));
    //
    // // Write particles into their buckets
    // for (uint32_t i = 0; i < src->particles_count; ++i) {
    //     Particle p = src->particles[i];
    //     if (particle_is_null(p)) continue;
    //
    //     uint32_t bucket_x = p.x / 1;
    //     uint32_t bucket_y = p.y / 1;
    //     uint32_t bucket = bucket_x * BUCKETS_X + bucket_y;
    //
    //     uint32_t last_idx = bucket_len[bucket]++;
    //     dst->particles[bucket * BUCKET_CAPACITY + last_idx] = p;
    // }
    //
    // // Write remaining slots as empty
    // for (uint32_t bucket = 0; bucket < BUCKETS_COUNT; ++bucket) {
    //     for (uint32_t i = bucket_len[bucket]; i < BUCKET_CAPACITY; ++i) {
    //         dst->particles[bucket * BUCKET_CAPACITY + i].ty = 0;
    //     }
    // }
}

static void kernel_init() {
    int gpus_count;
    cudaError_t error = cudaGetDeviceCount(&gpus_count);
    running_with_gpus = error == cudaSuccess && gpus_count > 0;

    size_t k_size = sizeof(Particle) * MAX_PARTICLE_COUNT;

    if (running_with_gpus) {
        cudaMalloc((void**)&k_0, k_size * 2);
        assert(k_0);

        cudaMalloc((void**)&k_1, k_size * 2);
        assert(k_1);

        cudaMalloc((void**)&k_internal, k_size * 2);
        assert(k_internal);

        cudaMallocHost((void**)&frame, packet_size(MAX_PARTICLE_COUNT) * 2);
        assert(frame);
    } else {
        frame = (FrameHeader*)malloc(packet_size(MAX_PARTICLE_COUNT) * 2);
        assert(frame);
    }

    *frame = frame_header_init();
}

static void kernel_destroy() {
    if (running_with_gpus) {
        cudaFree(k_0);
        cudaFree(k_1);
        cudaFree(k_internal);
        cudaFreeHost(frame);
    } else {
        free(frame);
    }
}

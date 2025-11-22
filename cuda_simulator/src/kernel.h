#pragma once
#include <assert.h>
#include <particle_io.h>
#include <cuda_runtime.h>
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

__global__ static void gpu_kernel(Particle* src, Particle* dst, float dt, uint32_t count) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    Particle p = src[i];

    dst[i].x = p.x + p.vx * dt;
    dst[i].y = p.y + p.vy * dt;
    dst[i].vx = p.vx;
    dst[i].vy = p.vy;
    dst[i].ty = p.ty;

    if (dst[i].x > 1.) dst[i].x = 0.;
    if (dst[i].y > 1.) dst[i].y = 0.;
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

static void step(Particle* src, Particle* dst, float dt, uint32_t count) {
    // cpu_kernel(src, dst, dt, count);

    uint32_t nThreads = 256;
    uint32_t nBlocks = (count + nThreads - 1) / nThreads;
    gpu_kernel<<<nBlocks, nThreads>>>(src, dst, dt, count);
}

static void run_kernel_async(FrameHeader* frame, Particle* k_src, Particle* k_dst) {
    const uint32_t steps = 1000 | 1;//frame->metadata.steps_per_frame | 1;

    float dt = frame->metadata.dt / steps;
    uint32_t count = frame->particles_count;

    step(k_src, k_dst, dt, count);
    for (uint32_t i = 1; i < steps; i += 2) {
        step(k_dst, k_internal, dt, count);
        step(k_internal, k_dst, dt, count);
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
    size_t k_size = sizeof(Particle) * MAX_PARTICLE_COUNT;

    cudaMalloc((void**)&k_0, k_size * 2);
    assert(k_0);

    cudaMalloc((void**)&k_1, k_size * 2);
    assert(k_1);

    cudaMalloc((void**)&k_internal, k_size * 2);
    assert(k_internal);

    cudaMallocHost((void**)&frame, packet_size(MAX_PARTICLE_COUNT) * 2);
    assert(frame);

    *frame = frame_header_init();
}

static void kernel_destroy() {
    cudaFree(k_0);
    cudaFree(k_1);
    cudaFree(k_internal);
    cudaFreeHost(frame);
}

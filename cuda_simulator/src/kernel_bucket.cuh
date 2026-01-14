#pragma once
#include "kernel.cuh"
#include "particle_io.h"

__host__ __device__ void bucket_move_kernel(const Particle* src, Particle* dst, FrameMetadata frame,
                                            uint32_t particle_count, uint32_t bucket_i) {

    uint32_t bucket_x = bucket_i % BUCKETS_X;
    uint32_t bucket_y = bucket_i / BUCKETS_X;

    int32_t x_min = bucket_x == 0 ? 0 : -1;
    int32_t x_max = bucket_x == BUCKETS_X - 1 ? 0 : 1;
    int32_t y_min = bucket_y == 0 ? 0 : -1;
    int32_t y_max = bucket_y == BUCKETS_Y - 1 ? 0 : 1;

    uint32_t i= 0;

    for (int32_t y = y_min; y <= y_max; ++y) {
        for (int32_t x = x_min; x <= x_max; ++x) {
            uint32_t bucket_j = ((x + bucket_x) + (y + bucket_y) * BUCKETS_Y) * BUCKET_CAPACITY;

            for (uint32_t jj = 0; jj < BUCKET_CAPACITY; ++jj) {
                uint32_t j = jj + bucket_j;
                if (src[j].ty < 0) continue;

                if (src[j].x >> (32 - BUCKETS_X_LOG2) != bucket_x ||
                    src[j].y >> (32 - BUCKETS_Y_LOG2) != bucket_y) continue;


                dst[bucket_i*BUCKET_CAPACITY + i++] = src[j];
                if (i == BUCKET_CAPACITY) return;
            }
        }
    }

    for (int32_t j = i; j < BUCKET_CAPACITY; ++j) {
        dst[bucket_i*BUCKET_CAPACITY + j].ty = -1;
    }
}
__host__ __device__ void bucket_step_kernel(const Particle* src, Particle* dst, FrameMetadata frame,
                                            uint32_t particle_count, uint32_t _i) {
    uint32_t i = 0;

    // i |= _i >> 4;
    // i |= (_i & 0b1111) << (16-4);

    i = _i;

    dst[i].ty = src[i].ty;
    if (src[i].ty < 0) return;

    const ParticleParams params(frame.particles[0]);

    float u32_max = (float)UINT32_MAX;
    float dx = frame.cursor_pos[0] - float(src[i].x) / u32_max;
    float dy = frame.cursor_pos[1] - float(src[i].y) / u32_max;

    float2 force = {0., 0.};
    float sq_dist = dx*dx + dy*dy;

    if (sq_dist < frame.cursor_size*frame.cursor_size/4) {
        force.x = 8e-12f / (sq_dist + 1.f);
        force.y = 8e-12f / (sq_dist + 1.f);

        if (dx > 0) force.x = -force.x;
        if (dy > 0) force.y = -force.y;
    }

    force += params.f_wall_force(src[i], frame);

    uint32_t bucket_x = (i / BUCKET_CAPACITY) % BUCKETS_X;
    uint32_t bucket_y = (i / BUCKET_CAPACITY) / BUCKETS_X;

    int32_t x_min = bucket_x == 0 ? 0 : -1;
    int32_t x_max = bucket_x == BUCKETS_X - 1 ? 0 : 1;
    int32_t y_min = bucket_y == 0 ? 0 : -1;
    int32_t y_max = bucket_y == BUCKETS_Y - 1 ? 0 : 1;

    for (int32_t y = y_min; y <= y_max; ++y) {
        for (int32_t x = x_min; x <= x_max; ++x) {
            uint32_t bucket_j = ((x + bucket_x) + (y + bucket_y) * BUCKETS_Y) * BUCKET_CAPACITY;

            for (uint32_t jj = 0; jj < BUCKET_CAPACITY; ++jj) {
                uint32_t j = jj + bucket_j;
                if (j == i || src[j].ty < 0) continue;

                float2 r = f_dist(src[i], src[j], frame);
                force += params.f2_force(r);
            }
        }
    }

    params.f_apply_force(dst[i], src[i], force, frame);
}

__global__ static void bucket_step_gpu(const Particle* src, Particle* dst, FrameMetadata frame,
                                       uint32_t count) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    bucket_step_kernel(src, dst, frame, count, i);
}

__global__ static void bucket_move_gpu(const Particle* src, Particle* dst, FrameMetadata frame,
                                       uint32_t count) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= BUCKETS_X*BUCKETS_Y) return;

    bucket_move_kernel(src, dst, frame, count, i);
}

static void bucket_step(const FrameHeader& frame, DeviceBufferId src_id, DeviceBufferId dst_id) {
    switch (frame.metadata.device) {
        case Device::Gpu: {
            Particle* d_src = kernel.buffer[src_id].gpu_particles;
            Particle* d_dst = kernel.buffer[dst_id].gpu_particles;

            uint32_t nThreads = 1 << frame.metadata.gpu_threads_per_block_log2;
            uint32_t nBlocks = (frame.particle_count + nThreads - 1) / nThreads;

            bucket_step_gpu<<<nBlocks, nThreads, 0, kernel.stream>>>(d_src, d_dst, frame.metadata,
                                                                     frame.particle_count);
            break;
        }
        case Device::CpuThreadPool: {
            Particle* d_src = kernel.buffer[src_id].cpu_particles;
            Particle* d_dst = kernel.buffer[dst_id].cpu_particles;

            kernel.pool.run((size_t)frame.particle_count, [=](size_t i) {
                bucket_step_kernel(d_src, d_dst, frame.metadata, frame.particle_count, i);
            });
            break;
        }
        case Device::CpuMainThread: {
            Particle* d_src = kernel.buffer[src_id].cpu_particles;
            Particle* d_dst = kernel.buffer[dst_id].cpu_particles;

            for (uint32_t i = 0; i < frame.particle_count; ++i) {
                bucket_step_kernel(d_src, d_dst, frame.metadata, frame.particle_count, i);
            }
            break;
        }
    }
}

static void bucket_move(const FrameHeader& frame, DeviceBufferId src_id, DeviceBufferId dst_id) {
    uint32_t n = BUCKETS_X * BUCKETS_Y;

    switch (frame.metadata.device) {
        case Device::Gpu: {
            Particle* d_src = kernel.buffer[src_id].gpu_particles;
            Particle* d_dst = kernel.buffer[dst_id].gpu_particles;

            uint32_t nThreads = 1 << frame.metadata.gpu_threads_per_block_log2;
            uint32_t nBlocks = (n + nThreads - 1) / nThreads;

            bucket_move_gpu<<<nBlocks, nThreads, 0, kernel.stream>>>(d_src, d_dst, frame.metadata,
                                                                     frame.particle_count);
            break;
        }
        case Device::CpuThreadPool: {
            Particle* d_src = kernel.buffer[src_id].cpu_particles;
            Particle* d_dst = kernel.buffer[dst_id].cpu_particles;

            kernel.pool.run((size_t)n, [=](size_t i) {
                bucket_move_kernel(d_src, d_dst, frame.metadata, frame.particle_count, i);
            });
            break;
        }
        case Device::CpuMainThread: {
            Particle* d_src = kernel.buffer[src_id].cpu_particles;
            Particle* d_dst = kernel.buffer[dst_id].cpu_particles;

            for (uint32_t i = 0; i < n; ++i) {
                bucket_move_kernel(d_src, d_dst, frame.metadata, frame.particle_count, i);
            }
            break;
        }
    }
}
static void bucket_kernel_run_async(DeviceBufferId d_src, DeviceBufferId d_dst) {
    const FrameHeader& frame = kernel.buffer[d_src].frame;

    const uint32_t move_every_n = 16;
    int32_t move_countdown = 0;
    uint32_t steps = 0;

    bucket_step(frame, d_src, d_dst);
    steps += 1;

    while (steps < frame.metadata.steps_per_frame) {
        if (move_countdown <= 0) {
            bucket_move(frame, d_dst, D_BUFFER_INTERNAL);
            move_countdown = move_every_n;

            bucket_step(frame, D_BUFFER_INTERNAL, d_dst);
            move_countdown -= 1;
            steps += 1;
        } else {
            bucket_step(frame, d_dst, D_BUFFER_INTERNAL);
            bucket_step(frame, D_BUFFER_INTERNAL, d_dst);
            move_countdown -= 2;
            steps += 2;
        }
    }
}

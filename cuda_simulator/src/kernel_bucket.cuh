#pragma once
#include "kernel.cuh"
#include "particle_io.h"

__host__ __device__ void bucket_step_kernel(const Particle* src, Particle* dst, FrameMetadata frame,
                                            uint32_t particle_count, uint32_t i) {
    dst[i].ty = src[i].ty;
    if (src[i].ty < 0) return;

    const ParticleParams params(frame.particles[0]);
    float2 force = {0., 0.};

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

static void bucket_step(const FrameHeader& frame, DeviceBufferId src_id, DeviceBufferId dst_id) {
    switch (frame.metadata.device) {
        case Device::Gpu: {
            Particle* d_src = kernel.buffer[src_id].gpu_particles;
            Particle* d_dst = kernel.buffer[dst_id].gpu_particles;

            uint32_t nThreads = 1 << frame.metadata.gpu_threads_per_block_log2;
            uint32_t nBlocks = (frame.particles_count + nThreads - 1) / nThreads;

            bucket_step_gpu<<<nBlocks, nThreads, 0, kernel.stream>>>(d_src, d_dst, frame.metadata,
                                                                     frame.particles_count);
            break;
        }
        case Device::CpuThreadPool: {
            Particle* d_src = kernel.buffer[src_id].cpu_particles;
            Particle* d_dst = kernel.buffer[dst_id].cpu_particles;

            kernel.pool.run((size_t)frame.particles_count, [=](size_t i) {
                bucket_step_kernel(d_src, d_dst, frame.metadata, frame.particles_count, i);
            });
            break;
        }
        case Device::CpuMainThread: {
            Particle* d_src = kernel.buffer[src_id].cpu_particles;
            Particle* d_dst = kernel.buffer[dst_id].cpu_particles;

            for (uint32_t i = 0; i < frame.particles_count; ++i) {
                bucket_step_kernel(d_src, d_dst, frame.metadata, frame.particles_count, i);
            }
            break;
        }
    }
}

static void bucket_kernel_run_async(DeviceBufferId d_src, DeviceBufferId d_dst) {
    const FrameHeader& frame = kernel.buffer[d_src].frame;

    if (frame.metadata.steps_per_frame % 2 == 0) {
        bucket_step(frame, d_src, D_BUFFER_INTERNAL);
        bucket_step(frame, D_BUFFER_INTERNAL, d_dst);
    } else {
        bucket_step(frame, d_src, d_dst);
    }

    for (uint32_t i = 2; i < frame.metadata.steps_per_frame; i += 2) {
        bucket_step(frame, d_dst, D_BUFFER_INTERNAL);
        bucket_step(frame, D_BUFFER_INTERNAL, d_dst);
    }
}

#pragma once
#include "kernel.cuh"

__host__ __device__ void compact_step_kernel(const Particle* src, Particle* dst,
                                             FrameMetadata frame, uint32_t particle_count,
                                             uint32_t i) {
    const ParticleParams params(frame.particles[0]);
    float2 force = {0., 0.};

    float u32_max = (float)UINT32_MAX;
    float dx = frame.cursor_pos[0] - float(src[i].x) / u32_max;
    float dy = frame.cursor_pos[1] - float(src[i].y) / u32_max;

    float sq_dist = dx*dx + dy*dy;

    if (sq_dist < frame.cursor_size*frame.cursor_size/4) {
        force.x = 8e-12f / (sq_dist + 1.f);
        force.y = 8e-12f / (sq_dist + 1.f);

        if (dx > 0) force.x = -force.x;
        if (dy > 0) force.y = -force.y;
    }

    force += params.f_wall_force(src[i], frame);

    for (uint32_t j = 0; j < particle_count; ++j) {
        if (j == i) continue;

        float2 r = f_dist(src[i], src[j], frame);
        force += params.f2_force(r);
    }

    params.f_apply_force(dst[i], src[i], force, frame);
}

__global__ static void compact_step_gpu(const Particle* src, Particle* dst, FrameMetadata frame,
                                        uint32_t count) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    compact_step_kernel(src, dst, frame, count, i);
}

static void compact_step(const FrameHeader& frame, DeviceBufferId src_id, DeviceBufferId dst_id) {
    switch (frame.metadata.device) {
        case Device::Gpu: {
            Particle* d_src = kernel.buffer[src_id].gpu_particles;
            Particle* d_dst = kernel.buffer[dst_id].gpu_particles;

            uint32_t nThreads = 1 << frame.metadata.gpu_threads_per_block_log2;
            uint32_t nBlocks = (frame.particle_count + nThreads - 1) / nThreads;

            compact_step_gpu<<<nBlocks, nThreads, 0, kernel.stream>>>(d_src, d_dst, frame.metadata,
                                                                      frame.particle_count);
            break;
        }
        case Device::CpuThreadPool: {
            Particle* d_src = kernel.buffer[src_id].cpu_particles;
            Particle* d_dst = kernel.buffer[dst_id].cpu_particles;

            kernel.pool.run((size_t)frame.particle_count, [=](size_t i) {
                compact_step_kernel(d_src, d_dst, frame.metadata, frame.particle_count, i);
            });
            break;
        }
        case Device::CpuMainThread: {
            Particle* d_src = kernel.buffer[src_id].cpu_particles;
            Particle* d_dst = kernel.buffer[dst_id].cpu_particles;

            for (uint32_t i = 0; i < frame.particle_count; ++i) {
                compact_step_kernel(d_src, d_dst, frame.metadata, frame.particle_count, i);
            }
            break;
        }
    }
}

static void compact_kernel_run_async(size_t d_src, size_t d_dst) {
    const FrameHeader& frame = kernel.buffer[d_src].frame;

    if (frame.metadata.steps_per_frame % 2 == 0) {
        compact_step(frame, d_src, D_BUFFER_INTERNAL);
        compact_step(frame, D_BUFFER_INTERNAL, d_dst);
    } else {
        compact_step(frame, d_src, d_dst);
    }

    for (uint32_t i = 2; i < frame.metadata.steps_per_frame; i += 2) {
        compact_step(frame, d_dst, D_BUFFER_INTERNAL);
        compact_step(frame, D_BUFFER_INTERNAL, d_dst);
    }
}

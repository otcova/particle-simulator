#pragma once
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "log.h"
#include "particle.h"
#include "particle_io.h"
#include "thread_pool.hpp"

#define BUCKET_CAPACITY 16
#define BUCKETS_X 32
#define BUCKETS_Y 32
#define BUCKETS_COUNT (BUCKETS_X * BUCKETS_Y)
#define MAX_PARTICLE_COUNT (BUCKET_CAPACITY * BUCKETS_COUNT)

#define K0 (MAX_PARTICLE_COUNT * 0)
#define K1 (MAX_PARTICLE_COUNT * 1)
#define KI (MAX_PARTICLE_COUNT * 2)

static Particle* gpu_buffer = NULL;
static Particle* cpu_buffer = NULL;
static FrameHeader* frame;

static int gpus_count;
static Device active_device = Device::CpuMainThread;

static ThreadPool pool;

// # Desmos
// C\ =\ \frac{n}{n-m}\left(\frac{n}{m}\right)^{\frac{m}{n-m}}
// F\left(r\right)
// =C\cdot p\cdot\frac{m\left(\frac{s}{r}\right)^{m}-n\left(\frac{s}{r}\right)^{n}}{r}
// V\left(r\right)
// =C\cdot p\left(\left(\frac{s}{r}\right)^{n}-\left(\frac{s}{r}\right)^{m}\right)
//
// Force Zero at:
// x=s\sqrt[n-m]{\frac{n}{m}}
//
// # Wolframalpha
// Partial[\(40)Power[\(40)Divide[s,x]\(41),n]-Power[\(40)Divide[s,x]\(41),m]\(41) ,x]

constexpr float k_b = 1.380649e-23;

__host__ __device__ void compact_kernel(const Particle* src, Particle* dst, FrameMetadata frame,
                                        uint32_t particle_count, uint32_t i) {
    float2 force = {0, 0};

    const ParticleParams params(frame.particles[0]);

    for (uint32_t j = 0; j < particle_count; ++j) {
        if (j == i) continue;

        float2 f = params.force(src[i], src[j]);
        force.x += f.x;
        force.y += f.y;
    }

    params.apply_force(dst[i], src[i], force, frame.step_dt);
    bounce_walls(dst[i], {frame.box_width, frame.box_height});
}

__global__ static void gpu_compact(const Particle* src, Particle* dst, FrameMetadata frame,
                                   uint32_t count) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    compact_kernel(src, dst, frame, count, i);
}

static void cpu_compact(const Particle* src, Particle* dst, FrameMetadata frame,
                        uint32_t particle_count) {
    if (frame.device == Device::CpuThreadPool) {
        pool.run((size_t)particle_count,
                 [=](size_t i) { compact_kernel(src, dst, frame, particle_count, i); });
    } else {
        for (uint32_t i = 0; i < particle_count; ++i) {
            compact_kernel(src, dst, frame, particle_count, i);
        }
    }
}

static void step(size_t src_offset, size_t dst_offset, const FrameHeader* frame) {
    if (frame->metadata.device == Device::Gpu) {
        Particle* src = gpu_buffer + src_offset;
        Particle* dst = gpu_buffer + dst_offset;

        uint32_t nThreads = 256;
        uint32_t nBlocks = (frame->particles_count + nThreads - 1) / nThreads;
        gpu_compact<<<nBlocks, nThreads>>>(src, dst, frame->metadata, frame->particles_count);
    } else {
        Particle* src = cpu_buffer + src_offset;
        Particle* dst = cpu_buffer + dst_offset;
        cpu_compact(src, dst, frame->metadata, frame->particles_count);
    }
}

static void kernel_run_async(const FrameHeader* frame, size_t k_src, size_t k_dst) {
    if (frame->metadata.steps_per_frame % 2 == 0) {
        step(k_src, KI, frame);
        step(KI, k_dst, frame);
    } else {
        step(k_src, k_dst, frame);
    }

    for (uint32_t i = 2; i < frame->metadata.steps_per_frame; i += 2) {
        step(k_dst, KI, frame);
        step(KI, k_dst, frame);
    }
}

static void kernel_sync(FrameHeader* frame) {
    if (frame->metadata.device == Device::Gpu) {
        // cudaSync(stream);
    } else if (frame->metadata.device == Device::CpuThreadPool) {
        pool.sync();
    }
}

// Convert a list of particles into the data-structure used by the kernel
static void kernel_prepare_frame(FrameHeader* src, FrameHeader* dst) {
    // Force Capabilities
    src->metadata.data_structure = DataStructure::CompactArray;
    if (src->metadata.device == Device::Gpu && gpus_count == 0) {
        src->metadata.device = Device::CpuThreadPool;
    }
    src->metadata.particles[0].epsilon *= k_b;
    src->metadata.particles[1].epsilon *= k_b;

    if (src->metadata.data_structure == DataStructure::CompactArray) {
        frame_compact_into(src, dst);
    } else if (src->metadata.data_structure == DataStructure::MatrixBuckets) {
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
        //
    }

    // Since both cpu device targets use the same buffers,
    // we have to wait for the async thread pool to stop using them.
    if (active_device == Device::CpuThreadPool && dst->metadata.device == Device::CpuMainThread) {
        pool.sync();
    }
    active_device = (Device)dst->metadata.device;
}

static void kernel_write(FrameHeader* src, size_t dst_offset) {
    size_t size = sizeof(Particle) * src->particles_count;
    if (src->metadata.device == Device::Gpu) {
        Particle* dst = gpu_buffer + dst_offset;
        cudaMemcpy(dst, &src->particles, size, cudaMemcpyHostToDevice);
    } else {
        Particle* dst = cpu_buffer + dst_offset;
        memcpy(dst, &src->particles, size);
    }
}

static void kernel_read(size_t src_offset, FrameHeader* dst) {
    size_t size = sizeof(Particle) * dst->particles_count;
    if (dst->metadata.device == Device::Gpu) {
        Particle* src = gpu_buffer + src_offset;
        cudaMemcpy(&dst->particles, src, size, cudaMemcpyDeviceToHost);
    } else {
        Particle* src = cpu_buffer + src_offset;
        memcpy(&dst->particles, src, size);
    }
}

static void kernel_init() {
    cudaError_t error = cudaGetDeviceCount(&gpus_count);
    if (error != cudaSuccess) gpus_count = 0;

    size_t kernel_buffer_size = sizeof(Particle) * MAX_PARTICLE_COUNT * 3;

    if (gpus_count > 0) {
        cudaMalloc((void**)&gpu_buffer, kernel_buffer_size);
        assert(gpu_buffer);

        cudaMallocHost((void**)&frame, packet_size(MAX_PARTICLE_COUNT));
        assert(frame);
    } else {
        frame = (FrameHeader*)malloc(packet_size(MAX_PARTICLE_COUNT));
        assert(frame);
    }

    cpu_buffer = (Particle*)malloc(kernel_buffer_size);
    assert(cpu_buffer);

    *frame = frame_header_init();
}

static void kernel_destroy() {
    pool.sync();

    if (gpus_count > 0) {
        cudaDeviceSynchronize();
        cudaFree(gpu_buffer);
        cudaFreeHost(frame);
    } else {
        free(frame);
    }

    free(cpu_buffer);
}

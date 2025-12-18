#pragma once

#include <cassert>
#include "lib/log.hpp"
#include "lib/thread_pool.hpp"
#include "particle.cuh"
#include "particle_io.h"

typedef uint32_t DeviceBufferId;
constexpr DeviceBufferId D_BUFFER_0 = 0;
constexpr DeviceBufferId D_BUFFER_1 = 1;
constexpr DeviceBufferId D_BUFFER_INTERNAL = 2;

#define BUCKET_CAPACITY 16
#define BUCKETS_X_LOG2 6
#define BUCKETS_Y_LOG2 6
#define BUCKETS_X (1 << BUCKETS_X_LOG2)
#define BUCKETS_Y (1 << BUCKETS_Y_LOG2)
#define BUCKETS_COUNT (BUCKETS_X * BUCKETS_Y)
#define MAX_PARTICLE_COUNT (BUCKET_CAPACITY * BUCKETS_COUNT)

struct Kernel {
    struct {
        Particle* cpu_particles;
        Particle* gpu_particles;
        FrameHeader frame;
    } buffer[3];

    FrameHeader* h_frame;  // Host frame

    ThreadPool pool;

    int gpus_count;
    Device active_device = Device::CpuMainThread;

    cudaStream_t stream = NULL;

    Kernel() {
        cudaError_t error = cudaGetDeviceCount(&gpus_count);
        if (error != cudaSuccess) gpus_count = 0;

<<<<<<< HEAD
    force += params.f_wall_force(src[i], frame);

    uint32_t bucket_x = (i / BUCKET_CAPACITY) % BUCKETS_X;
    uint32_t bucket_y = (i / BUCKET_CAPACITY) / BUCKETS_X;

    int32_t x_min = bucket_x == 0 ? 0 : -1;
    int32_t x_max = bucket_x == BUCKETS_X - 1? 0 : 1;
    int32_t y_min = bucket_y == 0 ? 0 : -1;
    int32_t y_max = bucket_y == BUCKETS_Y - 1? 0 : 1;

    for (int32_t y = y_min; y <= y_max; ++y) {
        for (int32_t x = x_min; x <= x_max; ++x) {
            uint32_t bucket_j = ((x + bucket_x) + (y+bucket_y) * BUCKETS_Y) * BUCKET_CAPACITY;
=======
        typedef Particle ParticleBuffer[3][MAX_PARTICLE_COUNT];
        ParticleBuffer* buffer_gpu = NULL;
        ParticleBuffer* buffer_cpu = NULL;
>>>>>>> 4500d99f078da12b4a539eda075eba378be2d91f

        if (gpus_count > 0) {
            cudaStreamCreate(&stream);
            assert(stream);

            cudaMalloc((void**)&buffer_gpu, sizeof(ParticleBuffer));
            assert(buffer_gpu);

            cudaMallocHost((void**)&h_frame, packet_size(MAX_PARTICLE_COUNT));
            assert(h_frame);
        } else {
            h_frame = (FrameHeader*)malloc(packet_size(MAX_PARTICLE_COUNT));
            assert(h_frame);
        }
<<<<<<< HEAD
    }
=======
>>>>>>> 4500d99f078da12b4a539eda075eba378be2d91f

        buffer_cpu = (ParticleBuffer*)malloc(sizeof(ParticleBuffer));
        assert(buffer_cpu);

        for (uint32_t i = 0; i < 3; ++i) {
            buffer[i].cpu_particles = &(*buffer_cpu)[i][0];
            buffer[i].gpu_particles = &(*buffer_gpu)[i][0];
        }

        *h_frame = frame_header_init();
        h_frame->particle_count = MAX_PARTICLE_COUNT;
    }

    ~Kernel() {
        pool.sync();

        if (gpus_count > 0) {
            cudaDeviceSynchronize();
            cudaStreamDestroy(stream);

            cudaFree(buffer[0].gpu_particles);
            cudaFreeHost(h_frame);
        } else {
            free(h_frame);
        }

        free(buffer[0].cpu_particles);
    }

    void sync() {
        if (h_frame->metadata.device == Device::Gpu) {
            cudaStreamSynchronize(stream);
        } else if (h_frame->metadata.device == Device::CpuThreadPool) {
            pool.sync();
        }
    }

    void write_metadata(DeviceBufferId dst_id) {
        FrameHeader const* src = h_frame;

        auto& dst = buffer[dst_id];
        dst.frame.metadata = src->metadata;
    }

    void write(DeviceBufferId dst_id) {
        FrameHeader const* src = h_frame;

        auto& dst = buffer[dst_id];
        dst.frame = *src;

        size_t size = sizeof(Particle) * src->particle_count;
        if (src->metadata.device == Device::Gpu) {
            cudaMemcpy(dst.gpu_particles, &src->particles, size, cudaMemcpyHostToDevice);
        } else {
            memcpy(dst.cpu_particles, &src->particles, size);
        }
    }

    void read(DeviceBufferId src_offset) {
        const auto& src = buffer[src_offset];
        FrameHeader* dst = h_frame;

        *dst = src.frame;

        size_t size = sizeof(Particle) * dst->particle_count;
        if (dst->metadata.device == Device::Gpu) {
            cudaMemcpy(&dst->particles, src.gpu_particles, size, cudaMemcpyDeviceToHost);
        } else {
            memcpy(&dst->particles, src.cpu_particles, size);
        }
    }

    void run_async(DeviceBufferId k_src, DeviceBufferId k_dst);
};

Kernel kernel;

#include "kernel_bucket.cuh"
#include "kernel_compact.cuh"

void Kernel::run_async(DeviceBufferId k_src, DeviceBufferId k_dst) {
    const FrameHeader& frame = kernel.buffer[k_src].frame;
    kernel.buffer[k_dst].frame = frame;

    switch (frame.metadata.data_structure) {
        case DataStructure::CompactArray:
            compact_kernel_run_async(k_src, k_dst);
            break;
        case DataStructure::MatrixBuckets:
            bucket_kernel_run_async(k_src, k_dst);
            break;
    }
}

void log_precision(const FrameHeader& frame) {
    ParticleParams p(frame.metadata.particles[0]);

    log("--- 0 Dist ---");

    double box_size = frame.metadata.box_width;
    double r0 = p.d_force0_r();
    const double u32_max = (double)UINT32_MAX;
    const double u64_max = (double)UINT64_MAX;

    double d_r = (r0 + box_size) - box_size;
    float f_r = ((float)r0 + (float)box_size) - (float)box_size;

    uint32_t u32_r = round((r0 / box_size) * u32_max);
    double d32_r = box_size * (double(u32_r) / u32_max);

    uint64_t u64_r = round((r0 / box_size) * u64_max);
    double d64_r = box_size * (double(u64_r) / u64_max);

    log("Ideal Float:  %.15e", (float)r0);
    log("Ideal Double: %.15e\n", r0);

    log("Box   Float:  %.15e", f_r);
    log("u32   Float:  %.15e", (float)d32_r);
    log("u32   Double: %.15e", d32_r);
    log("Box   Double: %.15e", d_r);
    log("u64   Double: %.15e", d64_r);

    log("--- Acc ---");

    log("Max Attraction:   %+e", p.d_force(p.d_max_attraction_r()) / p.mass);
    uint32_t div = BUCKETS_X / 2;
    log("1/%u box size     %+e\n", div, p.d_force(box_size / (double)div) / p.mass);

    log("Ideal Float  Mie: %+e", p.f_force(r0) / p.mass);
    log("Ideal Double Mie: %+e\n", p.d_force(r0) / p.mass);

    log("Box   Float  Mie: %+e", p.f_force(f_r) / p.mass);
    log("u32   Float  Mie: %+e", p.f_force(d32_r) / p.mass);
    log("u64   Float  Mie: %+e", p.f_force(d64_r) / p.mass);
    log("u32   Double Mie: %+e", p.d_force(d32_r) / p.mass);
    log("Box   Double Mie: %+e", p.d_force(d_r) / p.mass);
    log("u64   Double Mie: %+e", p.d_force(d64_r) / p.mass);
    log("---");
}

// Convert a list of particles into the data-structure used by the kernel
void kernel_prepare_frame(FrameHeader* src, FrameHeader* dst) {
    // Force Capabilities
    // src->metadata.data_structure = DataStructure::CompactArray;
    if (src->metadata.device == Device::Gpu && kernel.gpus_count == 0) {
        src->metadata.device = Device::CpuThreadPool;
    }

    if (src->metadata.data_structure == DataStructure::CompactArray) {
        dst->particle_count = MAX_PARTICLE_COUNT;  // dst capacity
        frame_compact_into(src, dst);
    } else if (src->metadata.data_structure == DataStructure::MatrixBuckets) {
        dst->metadata = src->metadata;

        if (src->particle_count > 0) {
            dst->particle_count = MAX_PARTICLE_COUNT;

            // Particle count per bucket
            uint32_t bucket_len[BUCKETS_X * BUCKETS_Y] = {0};

            // Write particles into their buckets
            for (uint32_t i = 0; i < src->particle_count; ++i) {
                Particle p = src->particles[i];
                if (p.ty < 0) continue;

                uint32_t bucket_x = p.x >> (32 - BUCKETS_X_LOG2);
                uint32_t bucket_y = p.y >> (32 - BUCKETS_Y_LOG2);
                uint32_t bucket = bucket_x + bucket_y * BUCKETS_X;

                uint32_t last_idx = bucket_len[bucket]++;
                dst->particles[bucket * BUCKET_CAPACITY + last_idx] = p;
            }

            // Write remaining empty slots
            for (uint32_t bucket = 0; bucket < BUCKETS_COUNT; ++bucket) {
                for (uint32_t i = bucket_len[bucket]; i < BUCKET_CAPACITY; ++i) {
                    dst->particles[bucket * BUCKET_CAPACITY + i].ty = -1;
                }
            }
        }
    }

    // Since both cpu device targets use the same buffers,
    // we have to wait for the async thread pool to stop using them.
    if (kernel.active_device == Device::CpuThreadPool &&
        dst->metadata.device == Device::CpuMainThread) {
        kernel.pool.sync();
    }
    kernel.active_device = (Device)dst->metadata.device;

    // log_precision(*dst);
}

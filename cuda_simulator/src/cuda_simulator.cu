#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <threads.h>
#include "frontend.h"
#include "kernel.h"

static void write_gpu(FrameHeader* src, Particle* dst) {
    size_t size = sizeof(Particle) * src->particles_count;
    // memcpy(dst, &src->particles, size);
    cudaMemcpy(dst, &src->particles, size, cudaMemcpyHostToDevice);
}

static void read_gpu(Particle* src, FrameHeader* dst) {
    size_t size = sizeof(Particle) * dst->particles_count;
    // memcpy(&dst->particles, src, size);
    cudaMemcpy(&dst->particles, src, size, cudaMemcpyDeviceToHost);
}

static void runtime(Particle* src, Particle* dst) {
    sync_kernel();
    run_kernel_async(frame, src, dst);

    if (receive_from_frontend(frame)) {
        write_gpu(frame, src);
        run_kernel_async(frame, src, dst);
        send_to_frontend(frame);
    } else {
        read_gpu(src, frame);
        send_to_frontend(frame);
    }
}

int main() {
    frontend_init_tcp();
    kernel_init();

    // Wait for first frame
    while (!receive_from_frontend(frame)) {
        thrd_yield();
    }

    write_gpu(frame, k_0);
    run_kernel_async(frame, k_0, k_1);
    send_to_frontend(frame);

    while (1) {
        runtime(k_1, k_0);
        runtime(k_0, k_1);
        // to not have to wait 1000 years to see changes
        // while (!receive_from_frontend(frame)) {
        //    thrd_yield();
        //}
        // write_gpu(frame, k_1);
    }

    kernel_destroy();
    frontend_destroy();
}

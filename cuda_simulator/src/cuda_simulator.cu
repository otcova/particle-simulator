#include <stdlib.h>
#include <string.h>
#include <threads.h>
#include "kernel.cuh"
#include "lib/frontend.hpp"

static Frontend frontend;

static void compute_frame(size_t src, size_t dst) {
    kernel_sync(frame);
    kernel_run_async(frame, src, dst);

    if (frontend.read(frame)) {
        kernel_write(frame, src);
        kernel_run_async(frame, src, dst);
        frontend.write(frame);
    } else {
        kernel_read(src, frame);
        frontend.write(frame);
    }
}

void main_loop() {
    if (!frontend.is_connected) return;

    kernel_write(frame, K0);
    if (!frontend.is_connected) return;

    kernel_run_async(frame, K0, K1);
    frontend.write(frame);

    while (1) {
        compute_frame(K1, K0);
        if (!frontend.is_connected) return;

        compute_frame(K0, K1);
        if (!frontend.is_connected) return;
    }
}

int main() {
    frontend.init_tcp();
    kernel_init();

    // Wait for first frame
    while (!frontend.read(frame) && frontend.is_connected) {
        thrd_yield();
    }

    main_loop();

    kernel_destroy();
}

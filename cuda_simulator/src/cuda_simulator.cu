#include <stdlib.h>
#include <string.h>
#include <threads.h>
#include "frontend.h"
#include "kernel.h"

static void compute_frame(size_t src, size_t dst) {
    kernel_sync(frame);
    kernel_run_async(frame, src, dst);

    if (frontend_read(frame)) {
        kernel_write(frame, src);
        kernel_run_async(frame, src, dst);
        frontend_write(frame);
    } else {
        kernel_read(src, frame);
        frontend_write(frame);
    }
}

void main_loop() {
    if (!frontend_is_connected) return;

    kernel_write(frame, K0);
    if (!frontend_is_connected) return;

    kernel_run_async(frame, K0, K1);
    frontend_write(frame);

    while (1) {
        compute_frame(K1, K0);
        if (!frontend_is_connected) return;

        compute_frame(K0, K1);
        if (!frontend_is_connected) return;
    }
}

int main() {
    frontend_init_tcp();
    kernel_init();

    // Wait for first frame
    while (!frontend_read(frame) && frontend_is_connected) {
        thrd_yield();
    }

    main_loop();

    kernel_destroy();
    frontend_destroy();
}

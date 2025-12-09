#include <thread>
#include "kernel.cuh"
#include "lib/frontend.hpp"

static Frontend frontend;

static void compute_frame(DeviceBufferId d_src, DeviceBufferId d_dst) {
    kernel.sync();
    kernel.run_async(d_src, d_dst);

    if (frontend.read(kernel.h_frame)) {
        if (kernel.h_frame->particle_count == 0) {
            // Interactive mode (only modify metadata)
            kernel.write_metadata(d_dst);
        } else {
            // Compute hole frame from scratch
            kernel.write(d_src);
            kernel.run_async(d_src, d_dst);
            frontend.write(kernel.h_frame);
            return;
        }
    }

    kernel.read(d_src);
    frontend.write(kernel.h_frame);
}

void main_loop() {
    kernel.write(D_BUFFER_0);
    kernel.run_async(D_BUFFER_0, D_BUFFER_1);
    frontend.write(kernel.h_frame);

    while (frontend.is_connected) {
        compute_frame(D_BUFFER_1, D_BUFFER_0);
        if (!frontend.is_connected) break;
        compute_frame(D_BUFFER_0, D_BUFFER_1);
    }
}

int main() {
    frontend.init_tcp();

    // Wait for first frame
    while (frontend.is_connected) {
        if (frontend.read(kernel.h_frame) && kernel.h_frame->particle_count > 0) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    if (frontend.is_connected) {
        main_loop();
    }
}

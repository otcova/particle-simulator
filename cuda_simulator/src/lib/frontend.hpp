#pragma once
#include <particle_io.h>

void kernel_prepare_frame(FrameHeader* src, FrameHeader* dst);

struct Frontend {
    Writer writer;
    Reader reader;

    bool is_connected = false;

    void init_files() {
        reader_open_file(&reader, "../simulation/backend_in.bin");
        writer_open_file(&writer, "../simulation/backend_out.bin");
        is_connected = true;
    }

    void init_tcp() {
        // new_tcp_client(&reader, &writer, "10.192.196.245:53123");
        new_tcp_client(&reader, &writer, "0.0.0.0:53123");
    }

    ~Frontend() {
        reader_destroy(&reader);
        writer_destroy(&writer);
    }

    // Returns true if received data
    bool read(FrameHeader* frame) {
        Frame received_frame;
        received_frame.ptr = nullptr;

        is_connected = reader_read_last(&reader, &received_frame);
        if (!received_frame.ptr) return false;

        kernel_prepare_frame(received_frame.ptr, frame);
        frame_print(frame);
        frame_destroy(&received_frame);
        return true;
    }

    void write(FrameHeader* frame) {
        frame_compact(frame);
        is_connected = writer_write(&writer, frame);
    }
};

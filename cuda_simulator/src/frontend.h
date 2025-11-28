#pragma once
#include <particle_io.h>
#include "kernel.h"

static Writer writer;
static Reader reader;

static void frontend_init_files() {
    reader_open_file(&reader, "../simulation/backend_in.bin");
    writer_open_file(&writer, "../simulation/backend_out.bin");
}

static void frontend_init_tcp() {
    new_tcp_client(&reader, &writer, "0.0.0.0:53123");
}

static void frontend_destroy() {
    reader_destroy(&reader);
    writer_destroy(&writer);
}

// Returns true if received data
static bool receive_from_frontend(FrameHeader* frame) {
    Frame received_frame = reader_read_last(&reader);
    if (!received_frame.ptr) return false;

    frame_prepare(received_frame.ptr, frame);
    frame_print(frame);
    frame_destroy(&received_frame);
    return true;
}

static void send_to_frontend(FrameHeader* frame) {
    frame_compact(frame);
    writer_write(&writer, frame);
}

#pragma once
#include <particle_io.h>
#include "kernel.h"

static Writer writer;
static Reader reader;

bool frontend_is_connected = false;

static void frontend_init_files() {
    reader_open_file(&reader, "../simulation/backend_in.bin");
    writer_open_file(&writer, "../simulation/backend_out.bin");
    frontend_is_connected = true;
}

static void frontend_init_tcp() {
    new_tcp_client(&reader, &writer, "10.192.196.245:53123");
}

static void frontend_destroy() {
    reader_destroy(&reader);
    writer_destroy(&writer);
}

// Returns true if received data
static bool receive_from_frontend(FrameHeader* frame) {
    Frame received_frame;
    received_frame.ptr = NULL;

    frontend_is_connected = reader_read_last(&reader, &received_frame);
    if (!received_frame.ptr) return false;

    frame_prepare(received_frame.ptr, frame);
    frame_print(frame);
    frame_destroy(&received_frame);
    return true;
}

static void send_to_frontend(FrameHeader* frame) {
    frame_compact(frame);
    frontend_is_connected = writer_write(&writer, frame);
}

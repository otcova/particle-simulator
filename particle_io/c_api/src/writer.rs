use std::{
    ffi::{CStr, c_char},
    ptr::drop_in_place,
};

use particle_io::FrameHeader;

use crate::ptr_as_frame;

#[repr(C)]
pub struct Writer {
    _raw: [u64; 2],
}
const _: () = assert!(size_of::<Writer>() == size_of::<particle_io::Writer>());
const _: () = assert!(align_of::<Writer>() == align_of::<particle_io::Writer>());

/// # Safety
/// 1. The provided writer pointer must have the needed capacity & alignment
/// 2. The provided path must be a valid null terminated utf-8 c string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn writer_open_file(writer: *mut Writer, path: *const c_char) {
    let writer = writer as *mut particle_io::Writer;

    let path = unsafe { CStr::from_ptr(path) };
    let path = path.to_str().expect("invalid UTF-8");
    unsafe {
        std::ptr::write(writer, particle_io::Writer::open_file(path).unwrap());
    }
}

/// # Safety
/// 1. The provided writer pointer must be initialized and not yet destroyed
#[unsafe(no_mangle)]
pub unsafe extern "C" fn writer_destroy(writer: *mut Writer) {
    let writer = writer as *mut particle_io::Writer;

    unsafe { drop_in_place(writer) };
}

/// Returns false if the operations did not succeed
///
/// # Safety
/// 1. The provided writer pointer must be initialized
/// 2. The provided writer pointer must be initialized
#[unsafe(no_mangle)]
pub unsafe extern "C" fn writer_write(writer: *mut Writer, frame: *mut FrameHeader) -> bool {
    let writer = writer as *mut particle_io::Writer;

    unsafe {
        ptr_as_frame(frame, |frame| {
            if let Err(err) = (&mut *writer).write(frame) {
                eprintln!("[particle_io_c::Writer] {}", err);
                false
            } else {
                true
            }
        })
    }
}

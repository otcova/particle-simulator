use crate::particle::*;
use std::{
    ffi::{CStr, c_char},
    ptr::drop_in_place,
};

#[repr(C)]
pub struct Reader {
    _raw: [u64; 2],
}
const _: () = assert!(size_of::<Reader>() == size_of::<particle_io::Reader>());
const _: () = assert!(align_of::<Reader>() == align_of::<particle_io::Reader>());

/// # Safety
/// 1. The provided reader pointer must have the needed capacity & alignment
/// 2. The provided path must be a valid null terminated utf-8 c string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn reader_open_file(reader: *mut Reader, path: *const c_char) {
    let reader = reader as *mut particle_io::Reader;

    let path = unsafe { CStr::from_ptr(path) };
    let path = path.to_str().expect("invalid UTF-8");
    unsafe {
        std::ptr::write(reader, particle_io::Reader::open_file(path).unwrap());
    }
}

/// # Safety
/// 1. The provided reader pointer must be initialized and not yet destroyed
#[unsafe(no_mangle)]
pub unsafe extern "C" fn reader_destroy(reader: *mut Reader) {
    let reader = reader as *mut particle_io::Reader;

    unsafe { drop_in_place(reader) };
}

/// # Safety
/// 1. The provided reader pointer must be initialized
#[unsafe(no_mangle)]
pub unsafe extern "C" fn reader_read(reader: *mut Reader) -> Frame {
    let reader = reader as *mut particle_io::Reader;
    let received = unsafe { &*reader }.read().unwrap();
    Frame::from(received)
}

/// Returns false if the operations did not succeed
///
/// # Safety
/// 1. The provided reader pointer must be initialized
#[unsafe(no_mangle)]
pub unsafe extern "C" fn reader_read_last(reader: *mut Reader, frame: *mut Frame) -> bool {
    let reader = reader as *mut particle_io::Reader;
    let mut succeed = true;
    let received = std::iter::from_fn(|| match unsafe { &*reader }.read() {
        Ok(frame) => frame,
        Err(()) => {
            succeed = false;
            None
        }
    });
    unsafe { *frame = Frame::from(received.last()) };
    succeed
}

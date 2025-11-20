use crate::*;
use std::ffi::{CStr, c_char};

/// # Safety
/// 1. The provided reader & writer pointer must have the needed capacity & alignment
/// 2. The provided path must be a valid null terminated utf-8 c string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn new_tcp_client(
    reader: *mut Reader,
    writer: *mut Writer,
    addr: *const c_char,
) {
    let reader = reader as *mut particle_io::Reader;
    let writer = writer as *mut particle_io::Writer;

    let addr = unsafe { CStr::from_ptr(addr) };
    let addr = addr.to_str().expect("invalid UTF-8");
    let (rx, tx) = particle_io::new_tcp_client(addr).unwrap();
    unsafe {
        std::ptr::write(reader, rx);
        std::ptr::write(writer, tx);
    }
}

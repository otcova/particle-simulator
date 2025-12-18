use crate::*;
use std::ffi::{CStr, c_char};

/// Returns false if the operations did not succeed
///
/// # Safety
/// 1. The provided reader & writer pointer must have the needed capacity & alignment
/// 2. The provided path must be a valid null terminated utf-8 c string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn new_tcp_client(
    reader: *mut Reader,
    writer: *mut Writer,
    addr: *const c_char,
) -> bool {
    let reader = reader as *mut particle_io::Reader;
    let writer = writer as *mut particle_io::Writer;

    let addr = unsafe { CStr::from_ptr(addr) };
    let addr = addr.to_str().expect("invalid UTF-8");
    let (rx, tx) = match particle_io::new_tcp_client(addr) {
        Ok((rx, tx)) => (rx, tx),
        Err(err) => {
            eprintln!("[particle_io_c::TCP] {}", err);
            return false;
        }
    };

    unsafe {
        std::ptr::write(reader, rx);
        std::ptr::write(writer, tx);
    }

    true
}

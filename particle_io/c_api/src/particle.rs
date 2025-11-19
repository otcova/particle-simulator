pub use particle_io::{FrameHeader, FrameMetadata, Particle};

#[repr(C)]
#[derive(Default)]
pub struct Frame {
    pub ptr: *mut FrameHeader,
    pub cap: usize,
    pub len: usize,
}

impl From<particle_io::Frame> for Frame {
    fn from(frame: particle_io::Frame) -> Frame {
        let bytes = frame.into_bytes();
        Frame {
            len: bytes.len(),
            cap: bytes.capacity(),
            ptr: &mut bytes.leak()[0] as *mut u8 as *mut FrameHeader,
        }
    }
}

impl From<Option<particle_io::Frame>> for Frame {
    fn from(frame: Option<particle_io::Frame>) -> Frame {
        if let Some(frame) = frame {
            Frame::from(frame)
        } else {
            Frame::null()
        }
    }
}

impl Frame {
    pub(crate) fn null() -> Frame {
        Frame {
            ptr: std::ptr::null_mut(),
            cap: 0,
            len: 0,
        }
    }
}

pub(crate) unsafe fn ptr_as_frame<F: FnOnce(&mut particle_io::Frame)>(ptr: *mut FrameHeader, f: F) {
    let particles_count = unsafe { &*ptr }.particles_count;
    let size = FrameHeader::packet_size(particles_count);
    // Safety: We do not mutate nor drop the Vec
    let bytes = unsafe { Vec::from_raw_parts(ptr as *mut u8, size, size) };
    let mut frame = particle_io::Frame::from_bytes(bytes);

    f(&mut frame);

    frame.into_bytes().leak();
}

/// This destructor sets and checks the internal `ptr` to null, so it can be called more than
/// once without failing.
/// # Safety
/// 1. The provided frame pointer must have been initialized or destroyed
#[unsafe(no_mangle)]
pub unsafe extern "C" fn frame_destroy(frame: *mut Frame) {
    let frame = unsafe { &mut *frame };
    if !frame.ptr.is_null() && frame.cap > 0 {
        // Reconstruct the Vec to drop it and free the memory
        unsafe { Vec::from_raw_parts(frame.ptr as *mut u8, frame.len, frame.cap) };
        frame.ptr = std::ptr::null_mut();
    }
}

/// # Safety
/// 1. The provided frame pointer must have been initialized
#[unsafe(no_mangle)]
pub unsafe extern "C" fn frame_print(frame: *mut FrameHeader) {
    unsafe {
        ptr_as_frame(frame, |frame| print!("{}", frame));
    }
}

/// # Safety
/// 1. The provided frame pointer must have been initialized
#[unsafe(no_mangle)]
pub unsafe extern "C" fn frame_compact(frame: *mut FrameHeader) {
    unsafe {
        ptr_as_frame(frame, |frame| frame.compact());
    }
}

/// # Safety
/// 1. The provided frame pointer must have been initialized
#[unsafe(no_mangle)]
pub unsafe extern "C" fn frame_compact_into(frame: *mut FrameHeader, dst: *mut FrameHeader) {
    unsafe {
        ptr_as_frame(frame, |frame| {
            ptr_as_frame(dst, |dst| frame.compact_into(dst))
        });
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn packet_size(particles_count: u32) -> usize {
    FrameHeader::packet_size(particles_count)
}

#[unsafe(no_mangle)]
pub extern "C" fn frame_header_init() -> FrameHeader {
    FrameHeader::new(FrameMetadata::default(), 0)
}

#[unsafe(no_mangle)]
pub extern "C" fn particle_is_null(particle: Particle) -> bool {
    particle.is_null()
}

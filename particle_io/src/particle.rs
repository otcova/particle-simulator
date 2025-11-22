use std::fmt::Display;

use bytemuck::{
    Pod, Zeroable, bytes_of, cast_slice, cast_slice_mut, checked::from_bytes_mut, from_bytes,
};

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod, PartialEq, Debug, Default)]
pub struct Particle {
    pub x: f32,
    pub y: f32,
    pub vx: f32,
    pub vy: f32,
    pub ty: u32,
}

impl Particle {
    pub fn is_null(self) -> bool {
        self.ty == 0
    }
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod, PartialEq, Default)]
pub struct FrameMetadata {
    pub step_dt: f32,
    pub steps_per_frame: u32,
}

#[derive(Clone, PartialEq)]
pub struct Frame(Vec<u8>);

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod, PartialEq, Default)]
pub struct FrameHeader {
    signature_start: [u8; 4],
    pub particles_count: u32,
    pub metadata: FrameMetadata,
    signature_end: [u8; 4],

    // This forces `FrameHeader` to have the needed
    // padding and alignment to be followed by a Particle array.
    particles: [Particle; 0],
}

impl FrameHeader {
    const SIGNATURE_START: [u8; 4] = [0x36, 0xbc, 0xe9, 0xbd];
    const SIGNATURE_END: [u8; 4] = [0xac, 0xc4, 0x12, 0xec];

    pub fn is_valid(&self) -> bool {
        self.signature_start == Self::SIGNATURE_START && self.signature_end == Self::SIGNATURE_END
    }

    pub fn set_invalid_signature(&mut self) {
        self.signature_start = [0; 4];
        self.signature_end = [0; 4];
    }

    pub fn set_valid_signature(&mut self) {
        self.signature_start = Self::SIGNATURE_START;
        self.signature_end = Self::SIGNATURE_END;
    }

    pub fn new(metadata: FrameMetadata, particles_count: u32) -> FrameHeader {
        FrameHeader {
            signature_start: Self::SIGNATURE_START,
            particles_count,
            metadata,
            signature_end: Self::SIGNATURE_END,
            particles: [],
        }
    }

    pub fn packet_size(particles_count: u32) -> usize {
        size_of::<FrameHeader>() + size_of::<Particle>() * particles_count as usize
    }
}

impl Default for Frame {
    fn default() -> Self {
        Self::new()
    }
}

impl Display for Frame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "--- Frame ---")?;
        if !self.header().is_valid() {
            writeln!(f, "  signature error")?;
        }
        let FrameMetadata {
            step_dt,
            steps_per_frame,
        } = *self.metadata();
        writeln!(f, "  step dt = {}", step_dt)?;
        writeln!(f, "  steps per frame = {}", steps_per_frame)?;
        writeln!(f, "  particle_count = {}", self.particles().len())?;
        for (i, p) in self.particles().iter().enumerate().take(5) {
            writeln!(
                f,
                "  [{}] = {{ x={}, y={}, vx={}, vy={}, ty={} }}",
                i, p.x, p.y, p.vx, p.vy, p.ty
            )?;
        }
        if !self.particles().len() > 5 {
            writeln!(f, "  ...")?;
        }
        writeln!(f, "-----------")?;
        Ok(())
    }
}

impl Frame {
    pub fn new() -> Frame {
        let header = FrameHeader::new(FrameMetadata::default(), 0);
        Frame(bytes_of(&header).to_vec())
    }

    pub fn from_header(header: FrameHeader) -> Frame {
        let size = FrameHeader::packet_size(header.particles_count);
        let mut bytes = Vec::with_capacity(size);
        bytes.extend_from_slice(bytes_of(&header));
        // Safety:
        // 1. Particles is bytemuck::Pod
        // 2. Vec has at least size capacity given it was created with Vec::with_capacity
        unsafe { bytes.set_len(size) };
        Frame(bytes)
    }

    pub fn from_bytes(bytes: Vec<u8>) -> Frame {
        assert!(bytes.len() >= size_of::<FrameHeader>());
        let frame = Frame(bytes);
        assert!(FrameHeader::packet_size(frame.header().particles_count) == frame.0.len());
        frame
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.0
    }

    pub fn bytes(&self) -> &[u8] {
        &self.0
    }

    pub fn header(&self) -> &FrameHeader {
        let bytes = &self.0[..size_of::<FrameHeader>()];
        from_bytes::<FrameHeader>(bytes)
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        let bytes = &mut self.0[..size_of::<FrameHeader>()];
        from_bytes_mut::<FrameHeader>(bytes)
    }

    pub fn metadata(&self) -> &FrameMetadata {
        &self.header().metadata
    }

    pub fn metadata_mut(&mut self) -> &mut FrameMetadata {
        &mut self.header_mut().metadata
    }

    pub fn particles(&self) -> &[Particle] {
        let bytes = &self.0[size_of::<FrameHeader>()..];
        cast_slice(bytes)
    }

    pub fn particles_mut(&mut self) -> &mut [Particle] {
        let bytes = &mut self.0[size_of::<FrameHeader>()..];
        cast_slice_mut(bytes)
    }

    pub fn compact(&mut self) {
        let particles = self.particles_mut();

        let mut left = 0;
        let mut right = particles.len();

        while left < right {
            if particles[left].is_null() {
                right -= 1;
                particles.swap(left, right);
            } else {
                left += 1;
            }
        }
    }

    /// Prevent an extra copy by compacting directly into another buffer
    pub fn compact_into(&self, dst: &mut Frame) {
        println!("A1");
        *dst.metadata_mut() = *self.metadata();
        println!("A2");
        dst.clear();
        println!("A3");
        for p in self.particles() {
            println!("  -");
            if !p.is_null() {
                dst.push(*p);
            }
        }
        println!("F");
    }

    pub fn clear(&mut self) {
        self.0.truncate(size_of::<FrameHeader>());
        self.header_mut().particles_count = 0;
    }

    pub fn push(&mut self, particle: Particle) {
        self.0.extend_from_slice(bytes_of(&particle));
        self.header_mut().particles_count += 1;
    }

    /// Reserves space for at least an `additional` number of particles.
    pub fn reserve(&mut self, additional: u32) {
        self.0.reserve(size_of::<Particle>() * additional as usize);
    }

    pub fn push_square(&mut self, size: u32) {
        self.reserve(size * size);

        for idx_x in 0..size {
            for idx_y in 0..size {
                let a = (idx_x + 509186523).wrapping_mul(3644126341);
                let b = (idx_y + 153252321).wrapping_mul(4235235234);
                let c = (idx_x + 621876523).wrapping_mul(4124124122);
                let d = (idx_y + 364373752).wrapping_mul(1423513984);

                let vx = ((a ^ b) % 1024) as f32 / 1024.;
                let vy = ((c ^ d) % 1024) as f32 / 1024.;

                let mut x = 0.5;
                let mut y = 0.5;

                if size > 1 {
                    // Coordinates from 0 to 1 (inclusive)
                    x = idx_x as f32 / (size - 1) as f32;
                    y = idx_y as f32 / (size - 1) as f32;
                }

                self.push(Particle {
                    x: (x + 0.5) * 0.5,
                    y: (y + 0.5) * 0.5,
                    vx: (vx - 0.5) * 1000000.,
                    vy: (vy - 0.5) * 1000000.,
                    ty: 1,
                });
            }
        }
    }
}

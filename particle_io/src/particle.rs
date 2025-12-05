use std::fmt::Display;

use bytemuck::{
    Pod, Zeroable, bytes_of, cast_slice, cast_slice_mut, checked::from_bytes_mut, from_bytes,
};

pub type Vec2 = vector2d::Vector2D<f64>;

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod, PartialEq, Debug, Default)]
pub struct Particle {
    pub x: u32,
    pub y: u32,
    pub vx: f32,
    pub vy: f32,
    pub ty: i32,
}

impl Particle {
    pub fn is_null(self) -> bool {
        self.ty < 0
    }
    pub fn pos_u32(self) -> [u32; 2] {
        [self.x, self.y]
    }
    pub fn vel_f32(self) -> [f32; 2] {
        [self.vx, self.vy]
    }
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod, PartialEq, Default, Debug)]
pub struct MiePotentialParams {
    // Distance (meters) at which V = 0
    pub sigma: f32,
    // Dispersion energy (J)
    pub epsilon: f32,
    pub n: f32,
    pub m: f32,
}

impl MiePotentialParams {
    pub fn force0_r(self) -> f64 {
        let n = self.n as f64;
        let m = self.m as f64;
        let sigma = self.sigma as f64;
        sigma * (n / m).powf(1. / (n - m))
    }
}

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum DataStructure {
    CompactArray,
    MatrixBuckets,
}

impl DataStructure {
    pub fn name(self) -> &'static str {
        match self {
            DataStructure::CompactArray => "Compact Array",
            DataStructure::MatrixBuckets => "Matrix Buckets",
        }
    }
}

impl TryFrom<u32> for DataStructure {
    type Error = ();
    fn try_from(value: u32) -> Result<Self, Self::Error> {
        use DataStructure::*;
        match value {
            x if x == CompactArray as u32 => Ok(CompactArray),
            x if x == MatrixBuckets as u32 => Ok(MatrixBuckets),
            _ => Err(()),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Device {
    Gpu,
    CpuThreadPool,
    CpuMainThread,
}

impl Device {
    pub fn name(self) -> &'static str {
        match self {
            Device::Gpu => "GPU",
            Device::CpuThreadPool => "CPU Thread Pool",
            Device::CpuMainThread => "CPU Main Thread",
        }
    }
}

impl TryFrom<u32> for Device {
    type Error = ();
    fn try_from(value: u32) -> Result<Self, Self::Error> {
        use Device::*;
        match value {
            x if x == Gpu as u32 => Ok(Gpu),
            x if x == CpuThreadPool as u32 => Ok(CpuThreadPool),
            x if x == CpuMainThread as u32 => Ok(CpuMainThread),
            _ => Err(()),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod, PartialEq, Debug)]
pub struct FrameMetadata {
    pub particles: [MiePotentialParams; 2],
    pub step_dt: f32,
    pub steps_per_frame: u32,

    pub box_width: f32,
    pub box_height: f32,

    pub data_structure: u32,
    pub device: u32,
    pub _padding: [u32; 2],
}

impl Default for FrameMetadata {
    fn default() -> FrameMetadata {
        let k_b = 1.380649e-23;

        FrameMetadata {
            step_dt: 50e-15,
            steps_per_frame: 1_000,
            box_width: 50e-9,
            box_height: 50e-9,
            data_structure: DataStructure::MatrixBuckets as u32,
            device: Device::Gpu as u32,
            particles: [
                MiePotentialParams {
                    // Nitrogen
                    sigma: 3.609e-10,
                    epsilon: 105.79 * k_b,
                    n: 14.08,
                    m: 6.,
                },
                MiePotentialParams {
                    // Argon
                    sigma: 3.404e-10,
                    epsilon: 117.84 * k_b,
                    n: 12.085,
                    m: 6.,
                },
            ],
            _padding: [0; _],
        }
    }
}

impl FrameMetadata {
    pub fn new_particle(&self, pos: impl Into<Vec2>, vel: impl Into<Vec2>, ty: i32) -> Particle {
        let pos = pos.into();
        let vel = vel.into();
        Particle {
            x: (u32::MAX as f64 * pos.x / self.box_width as f64).round() as u32,
            y: (u32::MAX as f64 * pos.y / self.box_height as f64).round() as u32,
            vx: vel.x as f32,
            vy: vel.y as f32,
            ty,
        }
    }

    pub fn box_size(&self) -> Vec2 {
        Vec2::new(self.box_width as f64, self.box_height as f64)
    }

    pub fn frame_dt(&self) -> f32 {
        self.step_dt * self.steps_per_frame as f32
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct Frame(Vec<u8>);

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod, PartialEq, Default, Debug)]
pub struct FrameHeader {
    signature_start: [u8; 4],
    pub particles_count: u32,
    pub metadata: FrameMetadata,
    signature_end: [u8; 4],
    _padding: u32,

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
            _padding: 0,
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
            box_width,
            box_height,
            ..
        } = *self.metadata();
        writeln!(f, "  step dt = {}", step_dt)?;
        writeln!(f, "  steps per frame = {}", steps_per_frame)?;
        writeln!(f, "  box size = ({}, {})", box_width, box_height)?;
        writeln!(f, "  ...")?;
        if self.particles().is_empty() {
            writeln!(f, "  particles[0] = {{}}")?;
        } else {
            writeln!(f, "  particles[{}] = {{", self.particles().len())?;
            for (i, p) in self.particles().iter().enumerate().take(5) {
                writeln!(
                    f,
                    "    [{}] = {{ x={:.2}%, y={:.2}%, vx={}, vy={}, ty={} }}",
                    i,
                    100. * p.x as f64 / u64::MAX as f64,
                    100. * p.y as f64 / u64::MAX as f64,
                    p.vx,
                    p.vy,
                    p.ty
                )?;
            }
            if self.particles().len() > 5 {
                writeln!(f, "    ...")?;
            }
            writeln!(f, "  }}")?;
        }
        writeln!(f, "-------------")?;
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
        *dst.metadata_mut() = *self.metadata();
        dst.clear();
        for p in self.particles() {
            if !p.is_null() {
                dst.push(*p);
            }
        }
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
}

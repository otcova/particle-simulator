use std::{
    fs::OpenOptions,
    io::{self, Read, Write},
};

#[derive(Default)]
pub struct Packet {
    pub time: f32,
    pub particles: Box<[Particle]>,
}

impl Packet {
    pub fn square(size: u32) -> Packet {
        let mut particles = Vec::new();
        if size > 0 {
            particles.reserve_exact((size * size) as usize);

            for idx_x in 0..size {
                for idx_y in 0..size {
                    let x: f32;
                    let y: f32;

                    if size > 1 {
                        x = idx_x as f32 / (size - 1) as f32;
                        y = idx_y as f32 / (size - 1) as f32;
                    } else {
                        x = 0.5;
                        y = 0.5;
                    }

                    particles.push(Particle {
                        pos_x: (x + 0.5) * 0.5,
                        pos_y: (y + 0.5) * 0.5,
                        vel_x: 0.,
                        vel_y: 0.,
                    })
                }
            }
        }
        Packet {
            time: 0.,
            particles: particles.into_boxed_slice(),
        }
    }
}

pub struct Backend {
    backend_out: Option<Box<dyn Read>>,
    backend_in: Option<Box<dyn Write>>,
    backend_out_status: BackendStatus,
    backend_in_status: BackendStatus,

    packet_reader: PacketReader,
    write_buffer: Vec<u8>,
}

pub struct BackendStatus {
    pub state: BackendState,
    pub details: String,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum BackendState {
    Disconnected,
    Connected,
}

impl Backend {
    pub fn new() -> Backend {
        let mut backend = Backend {
            backend_out: None,
            backend_in: None,
            backend_out_status: BackendStatus {
                state: BackendState::Disconnected,
                details: "".into(),
            },
            backend_in_status: BackendStatus {
                state: BackendState::Disconnected,
                details: "".into(),
            },

            packet_reader: PacketReader::default(),
            write_buffer: Vec::new(),
        };
        backend.open_backend_files();
        backend
    }

    pub fn close_connection(&mut self) {
        self.backend_out = None;
        self.backend_in = None;

        self.backend_in_status.state = BackendState::Disconnected;
        self.backend_out_status.state = BackendState::Disconnected;

        self.backend_in_status.details = "Connection closed".into();
        self.backend_out_status.details = "Connection closed".into();
    }

    pub fn open_backend_files(&mut self) {
        self.packet_reader.clear();

        let out_name = "./backend_out.bin".into();
        let in_name = "./backend_in.bin".into();

        let backend_out = OpenOptions::new().read(true).open(&out_name);
        let backend_in = OpenOptions::new().append(true).open(&in_name);

        match &backend_out {
            Ok(_) => {
                self.backend_out_status.state = BackendState::Connected;
                self.backend_out_status.details = out_name;
            }
            Err(error) => {
                self.backend_out_status.state = BackendState::Disconnected;
                self.backend_out_status.details = format!("{}", error);
            }
        };
        match &backend_in {
            Ok(_) => {
                self.backend_in_status.state = BackendState::Connected;
                self.backend_in_status.details = in_name;
            }
            Err(error) => {
                self.backend_in_status.state = BackendState::Disconnected;
                self.backend_in_status.details = format!("{}", error);
            }
        };

        self.backend_out = backend_out.ok().map(|f| Box::new(f) as _);
        self.backend_in = backend_in.ok().map(|f| Box::new(f) as _);
    }

    pub fn backend_out_status(&mut self) -> &BackendStatus {
        &self.backend_out_status
    }

    pub fn backend_in_status(&mut self) -> &BackendStatus {
        &self.backend_in_status
    }

    pub fn load(&mut self, packets: &mut Vec<Packet>) {
        let Some(backend_out) = &mut self.backend_out else {
            return;
        };

        match self.packet_reader.read(backend_out, packets) {
            Ok(_) => {}
            Err(err) if err.kind() == io::ErrorKind::Interrupted => {}
            Err(_) => self.backend_out = None,
        }
    }

    pub fn store(&mut self, packet: &Packet) {
        let Some(backend_in) = &mut self.backend_in else {
            return;
        };

        self.write_buffer.clear();
        self.write_buffer
            .extend_from_slice(bytemuck::bytes_of(&PacketHeader::from(packet)));
        self.write_buffer
            .extend_from_slice(bytemuck::cast_slice(&packet.particles));

        if backend_in.write_all(&self.write_buffer).is_err() {
            self.backend_in = None;
        }
        self.write_buffer.clear();
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Zeroable, bytemuck::Pod)]
pub struct Particle {
    pub pos_x: f32,
    pub pos_y: f32,
    pub vel_x: f32,
    pub vel_y: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Zeroable, bytemuck::Pod)]
struct PacketHeader {
    signature: [u8; 8],
    time: f32,
    particles_count: u32,
}

impl PacketHeader {
    // Each header must start with this SIGNATURE.
    const SIGNATURE: [u8; 8] = 0xec12c4acbde9bc36_u64.to_le_bytes();
}

impl From<&Packet> for PacketHeader {
    fn from(packet: &Packet) -> Self {
        PacketHeader {
            signature: Self::SIGNATURE,
            time: packet.time,
            particles_count: packet.particles.len() as u32,
        }
    }
}

#[derive(Default)]
struct PacketReader {
    buffer: Vec<u8>,
}

impl PacketReader {
    fn clear(&mut self) {
        self.buffer.clear();
    }

    fn read<R: Read>(&mut self, r: &mut R, out: &mut Vec<Packet>) -> io::Result<()> {
        r.read_to_end(&mut self.buffer)?;
        while self.buffer.len() >= PacketHeader::SIGNATURE.len() + size_of::<PacketHeader>() {
            // Find signature
            let packet_start = self
                .buffer
                .windows(8)
                .position(|w| w == PacketHeader::SIGNATURE);
            let Some(packet_start) = packet_start else {
                self.buffer
                    .drain(..self.buffer.len() - PacketHeader::SIGNATURE.len());
                break;
            };

            // Do we have enough bytes for full header?
            let header_end = packet_start + size_of::<PacketHeader>();
            if self.buffer.len() < header_end {
                self.buffer.drain(..packet_start);
                break;
            };

            // Cast the header bytes
            let raw_header = &self.buffer[packet_start..header_end];
            let header: &PacketHeader = bytemuck::from_bytes(raw_header);

            // Do we have enough bytes for full packet?
            let packet_end = header_end + size_of::<Particle>() * header.particles_count as usize;
            if self.buffer.len() < packet_end {
                self.buffer.drain(..packet_start);
                break;
            };

            // Cast the particles bytes
            let raw_particles = &self.buffer[header_end..packet_end];
            let particles: &[Particle] = bytemuck::cast_slice(raw_particles);

            out.push(Packet {
                time: header.time,
                particles: Box::from(particles),
            });

            // Remove bytes including this header
            self.buffer.drain(..packet_end);
        }

        Ok(())
    }
}

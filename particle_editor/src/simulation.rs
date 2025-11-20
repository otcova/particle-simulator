use crate::backend::Backend;
use particle_io::Frame;

pub struct Simulation {
    timeline: Vec<Frame>,
    default_frame: Frame, // Returned when timeline is empty
    timeline_ram: usize,
}

impl Simulation {
    pub fn new() -> Simulation {
        let default_frame = Frame::new();

        Simulation {
            timeline: Vec::new(),
            default_frame,
            timeline_ram: 0,
        }
    }

    pub fn update(&mut self, backend: &mut Backend) {
        while let Some(frame) = backend.read() {
            self.timeline_ram += frame.bytes().len();
            self.timeline.push(frame);
        }
    }

    pub fn clear(&mut self) {
        self.timeline.clear();
        self.timeline_ram = 0;
    }

    pub fn timeline_frames_count(&mut self) -> u32 {
        self.timeline.len() as u32
    }

    pub fn frame(&mut self, idx: u32) -> &Frame {
        if self.timeline.is_empty() {
            return &self.default_frame;
        }
        let max_idx = self.timeline.len() - 1;
        &self.timeline[max_idx.min(idx as usize)]
    }

    pub fn timeline_ram(&self) -> usize {
        self.timeline_ram
    }
}

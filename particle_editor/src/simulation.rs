use crate::backend::Backend;
use particle_io::Frame;

pub struct Simulation {
    timeline: Vec<Frame>,
    blank: Frame, // Returned when timeline is empty
}

impl Simulation {
    pub fn new() -> Simulation {
        Simulation {
            timeline: Vec::new(),
            blank: Frame::new(),
        }
    }

    pub fn update(&mut self, backend: &mut Backend) {
        while let Some(frame) = backend.read() {
            self.timeline.push(frame);
        }
    }

    pub fn clear(&mut self) {
        self.timeline.clear();
    }

    pub fn timeline_frames_count(&mut self) -> u32 {
        self.timeline.len() as u32
    }

    pub fn frame(&mut self, idx: u32) -> &Frame {
        if self.timeline.is_empty() {
            return &self.blank;
        }
        let max_idx = self.timeline.len() - 1;
        &self.timeline[max_idx.min(idx as usize)]
    }
}

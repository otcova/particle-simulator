use crate::backend::{Backend, Packet};

pub struct Simulation {
    timeline: Vec<Packet>,
    blank: Packet, // Returned when timeline is empty
}

impl Simulation {
    pub fn new() -> Simulation {
        Simulation {
            timeline: Vec::new(),
            blank: Packet::default(),
        }
    }

    pub fn update(&mut self, backend: &mut Backend) {
        backend.load(&mut self.timeline);
    }

    pub fn clear(&mut self) {
        self.timeline.clear();
    }

    pub fn timeline_frames_count(&mut self) -> u32 {
        self.timeline.len() as u32
    }

    pub fn frame(&mut self, idx: u32) -> &Packet {
        if self.timeline.is_empty() {
            return &self.blank;
        }
        let max_idx = self.timeline.len() - 1;
        &self.timeline[max_idx.min(idx as usize)]
    }
}

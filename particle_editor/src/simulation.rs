use std::fmt::Debug;

use crate::backend::Backend;
use particle_io::Frame;

struct TimeInterval {
    start_time: f32,
    first_frame_ind: usize,
    dt: f32,
    frame_count: usize,
}

impl TimeInterval {
    pub fn add_frame(&mut self) {
        self.frame_count += 1;
    }

    pub fn get_frame_ind(&self, moment: f32) -> usize {
        let diff = moment - self.start_time;
        let diff_step = diff / self.dt;
        let ind = diff_step.round() as usize;
        ind + self.first_frame_ind
    }

    pub fn duration(&self) -> f32 {
        self.frame_count as f32 * self.dt
    }
}

impl Debug for TimeInterval {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "st_time:{}, f_frame:{}, dt:{}, f_cnt:{}",
            self.start_time, self.first_frame_ind, self.dt, self.frame_count
        )
    }
}

pub struct Simulation {
    frames: Vec<Frame>,
    times: Vec<TimeInterval>,
    default_frame: Frame, // Returned when timeline is empty
    timeline_ram: usize,
}

impl Simulation {
    pub fn new() -> Simulation {
        let default_frame = Frame::new();

        Simulation {
            frames: Vec::new(),
            times: Vec::new(),
            default_frame,
            timeline_ram: 0,
        }
    }

    pub fn update(&mut self, backend: &mut Backend) {
        while let Some(frame) = backend.read() {
            self.timeline_ram += frame.bytes().len();

            let f_dt = frame.metadata().step_dt;
            #[allow(clippy::needless_late_init)]
            let cur_time: f32;

            let t_i = self.times.last_mut();
            match t_i {
                Some(t_i) => {
                    if t_i.dt == f_dt {
                        t_i.add_frame();
                        self.frames.push(frame);
                        return;
                    }
                    cur_time = t_i.start_time + t_i.duration();
                }
                None => cur_time = 0.,
            }

            self.times.push(TimeInterval {
                start_time: cur_time,
                first_frame_ind: self.frames.len(),
                dt: f_dt,
                frame_count: 1,
            });
            self.frames.push(frame);
        }
    }

    pub fn clear(&mut self) {
        self.frames.clear();
        self.times.clear();
        self.timeline_ram = 0;
    }

    pub fn frame(&mut self, moment: f32) -> &Frame {
        #[allow(clippy::collapsible_if)]
        if !self.frames.is_empty() {
            if let Some(idx) = self.find_frame_ind(moment) {
                let max_idx = self.frames.len() - 1;
                return &self.frames[max_idx.min(idx)];
            }
        }
        &self.default_frame
    }

    pub fn timeline_ram(&self) -> usize {
        self.timeline_ram
    }

    fn find_frame_ind(&self, moment: f32) -> Option<usize> {
        for t_i in self.times.iter().rev() {
            if t_i.start_time <= moment {
                return Some(t_i.get_frame_ind(moment));
            }
        }
        None
    }

    pub fn sim_len(&self) -> f32 {
        let t_i = self.times.last();
        match t_i {
            Some(t_i) => t_i.start_time + t_i.duration(),
            None => 0.,
        }
    }

    pub fn print(&self, moment: f32) -> Vec<String> {
        let mut res = Vec::new();
        res.push(
            self.times
                .iter()
                .map(|t| format!("{:?} | ", t))
                .collect::<String>(),
        );
        let ind = self.find_frame_ind(moment);

        res.push(ind.unwrap_or(0xFFFFFF).to_string());
        res.push(
            (self.frames.len() - 1)
                .min(ind.unwrap_or(0xFFFFFF))
                .to_string(),
        );

        res
    }
}

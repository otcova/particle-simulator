use std::fmt::Debug;

use crate::backend::Backend;
use particle_io::Frame;

pub struct TimelineFrame<'a> {
    pub frame: &'a Frame,
    pub frame_time: f32,
    pub frame_index: usize,
}

#[derive(Clone, Copy)]
struct TimeInterval {
    start_time: f32,
    dt: f32,
    start_index: usize,
    // Invariant: NonZero<usize>
    frame_count: usize,
}

impl TimeInterval {
    pub fn frame_index(&self, time: f32) -> usize {
        let count = (time - self.start_time) / self.dt;
        let index = count.round() as isize;
        self.start_index + index.clamp(0, self.frame_count as isize - 1) as usize
    }

    pub fn frame_time(&self, global_frame_index: usize) -> f32 {
        let idx = (global_frame_index - self.start_index).min(self.frame_count - 1);
        self.start_time + self.dt * idx as f32
    }

    pub fn last_frame_index(&self) -> usize {
        self.start_index + self.frame_count - 1
    }

    pub fn end_time(&self) -> f32 {
        self.start_time + self.dt * (self.frame_count - 1) as f32
    }

    pub fn duration(&self) -> f32 {
        self.frame_count as f32 * self.dt
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
            self.push_frame(frame);
        }
    }

    fn push_frame(&mut self, frame: Frame) {
        let index = self.frames.len();
        let dt = frame.metadata().frame_dt();

        self.timeline_ram += frame.bytes().len();
        self.frames.push(frame);

        let start_time = match self.times.last_mut() {
            Some(interval) if interval.dt == dt => {
                interval.frame_count += 1;
                return;
            }
            Some(last_interval) => last_interval.start_time + last_interval.duration(),
            None => 0.,
        };

        self.times.push(TimeInterval {
            start_time,
            dt,
            start_index: index,
            frame_count: 1,
        });
    }

    pub fn clear(&mut self) {
        self.frames.clear();
        self.times.clear();
        self.timeline_ram = 0;
    }

    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    pub fn frame(&mut self, moment: f32) -> TimelineFrame<'_> {
        let (frame_index, frame_time) = self.find_frame_index(moment);
        TimelineFrame {
            frame: self.frames.get(frame_index).unwrap_or(&self.default_frame),
            frame_time,
            frame_index,
        }
    }

    pub fn timeline_ram(&self) -> usize {
        self.timeline_ram
    }

    // Returns (index, actual_frame_time)
    fn find_frame_index(&self, time: f32) -> (usize, f32) {
        let interval_index = match self
            .times
            .binary_search_by(|interval| interval.start_time.total_cmp(&time))
        {
            // Case time is <= 0 or we do not have frames
            Ok(0) | Err(0) => return (0, 0.),
            Ok(idx) => idx,
            Err(idx) => idx - 1,
        };

        let interval = self.times.get(interval_index).copied();
        let interval = interval.unwrap_or(self.times[self.times.len() - 1]);
        let next_interval = self.times.get(interval_index + 1);

        // Case time is in interval
        if time <= interval.end_time() || next_interval.is_none() {
            let frame_index = interval.frame_index(time);
            return (frame_index, interval.frame_time(frame_index));
        }

        // Case time is inbetween two intervals
        let next_interval = next_interval.unwrap();
        if time - interval.end_time() > next_interval.start_time - time {
            (interval.last_frame_index(), interval.end_time())
        } else {
            (next_interval.start_index, next_interval.start_time)
        }
    }

    pub fn sim_len(&self) -> f32 {
        match self.times.last() {
            Some(interval) => interval.end_time(),
            None => 0.,
        }
    }
}

impl Debug for Simulation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "-- Timeline (frame_count: {}) --", self.frame_count())?;
        for interval in &self.times {
            writeln!(f, "{:?}", interval)?;
        }
        writeln!(f, "--------------")
    }
}

impl Debug for TimeInterval {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "start_time: {:?}, dt: {:?}, start_index: {}, frame_count: {}",
            self.start_time, self.dt, self.start_index, self.frame_count
        )
    }
}

use rand::{Rng, distr::uniform::SampleRange};

use crate::*;
use core::f32;
use std::ops::RangeInclusive;

pub struct ParticleLattice {
    pub particle_count: (u32, u32),

    /// Particle Distance = sigma * factor
    pub distance_factor: f32,

    pub velocity: RangeInclusive<f32>,
}

impl ParticleLattice {
    pub fn hex_square(&self, frame: &mut Frame) {
        let total_particle_count = self.particle_count.0 * self.particle_count.1;
        if total_particle_count == 0 {
            return;
        }

        // Center x, y
        let cx = frame.metadata().box_width / 2.;
        let cy = frame.metadata().box_height / 2.;

        // Particle Radius x, y
        let rx = frame.metadata().particles[0].force0_r() * self.distance_factor;
        let ry = f32::sin(f32::consts::PI / 3.) * rx;

        // Start x, y
        let sx = cx - rx * (self.particle_count.0 - 1) as f32 / 2.;
        let sy = cy - ry * (self.particle_count.1 - 1) as f32 / 2.;

        let rng = &mut rand::rng();

        frame.reserve(total_particle_count);
        for idx_x in 0..self.particle_count.0 {
            for idx_y in 0..self.particle_count.1 {
                let v = self.velocity.clone().sample_single(rng).unwrap_or(0.);
                let angle = rng.random_range(0.0..2.0 * std::f32::consts::PI);
                let (ax, ay) = angle.sin_cos();

                let offset = if idx_y % 2 == 0 { 0. } else { rx / 2. };

                frame.push(Particle {
                    x: sx + rx * idx_x as f32 + offset,
                    y: sy + ry * idx_y as f32,
                    vx: v * ax,
                    vy: v * ay,
                    ty: 1,
                });
            }
        }
    }

    pub fn square(&self, frame: &mut Frame) {
        let total_particle_count = self.particle_count.0 * self.particle_count.1;
        if total_particle_count == 0 {
            return;
        }

        // Center x, y
        let cx = frame.metadata().box_width / 2.;
        let cy = frame.metadata().box_height / 2.;

        // Particle Radius x, y
        let r = frame.metadata().particles[0].force0_r() * self.distance_factor;

        // Start x, y
        let sx = cx - r * (self.particle_count.0 - 1) as f32 / 2.;
        let sy = cy - r * (self.particle_count.1 - 1) as f32 / 2.;

        let rng = &mut rand::rng();

        frame.reserve(total_particle_count);
        for idx_x in 0..self.particle_count.0 {
            for idx_y in 0..self.particle_count.1 {
                let v = self.velocity.clone().sample_single(rng).unwrap_or(0.);
                let angle = rng.random_range(0.0..2.0 * std::f32::consts::PI);
                let (ax, ay) = angle.sin_cos();

                frame.push(Particle {
                    x: sx + r * idx_x as f32,
                    y: sy + r * idx_y as f32,
                    vx: v * ax,
                    vy: v * ay,
                    ty: 1,
                });
            }
        }
    }
}

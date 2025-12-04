use crate::*;
use core::{f32, f64};
use rand::{Rng, distr::uniform::SampleRange, rngs::ThreadRng};
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

        let meta = *frame.metadata();

        // Particle Radius x, y
        let rx = meta.particles[0].force0_r() * self.distance_factor as f64;
        let ry = f64::sin(f64::consts::PI / 3.) * rx;

        let center = meta.box_size() / 2.;
        let start = center
            - Vec2::new(
                rx * (self.particle_count.0 - 1) as f64 / 2.,
                ry * (self.particle_count.1 - 1) as f64 / 2.,
            );

        let rng = &mut rand::rng();

        frame.reserve(total_particle_count);
        for idx_x in 0..self.particle_count.0 {
            for idx_y in 0..self.particle_count.1 {
                let offset = if idx_y % 2 == 0 { 0. } else { rx / 2. };

                let pos = start + Vec2::new(rx * idx_x as f64 + offset, ry * idx_y as f64);
                let vel = self.random_vel(rng);
                frame.push(meta.new_particle(pos, vel, 0));
            }
        }
    }

    pub fn square(&self, frame: &mut Frame) {
        let total_particle_count = self.particle_count.0 * self.particle_count.1;
        if total_particle_count == 0 {
            return;
        }

        let meta = *frame.metadata();

        let center = meta.box_size() / 2.;
        let r = meta.particles[0].force0_r() * self.distance_factor as f64;
        let start = center
            - Vec2::new(
                (self.particle_count.0 - 1) as f64 / 2.,
                (self.particle_count.1 - 1) as f64 / 2.,
            ) * r;

        let rng = &mut rand::rng();

        frame.reserve(total_particle_count);
        for idx_x in 0..self.particle_count.0 {
            for idx_y in 0..self.particle_count.1 {
                let pos = start + Vec2::new(idx_x as f64, idx_y as f64) * r;
                let vel = self.random_vel(rng);
                frame.push(meta.new_particle(pos, vel, 0));
            }
        }
    }

    fn random_vel(&self, rng: &mut ThreadRng) -> Vec2 {
        let v = self.velocity.clone().sample_single(rng).unwrap_or(0.);
        let angle = rng.random_range(0.0..2.0 * f32::consts::PI);
        let dir = Vec2::from(angle.sin_cos());
        dir * v as f64
    }
}

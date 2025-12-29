use crate::*;
use core::{f32, f64};
use postcard::{from_bytes, to_vec};
use rand::{Rng, distr::uniform::SampleRange, rngs::ThreadRng};
use serde::{Deserialize, Serialize};
use std::{fs, io::Read, ops::RangeInclusive, path::Path};

pub struct ParticleLattice {
    pub particle_count: (u32, u32),

    /// Particle Distance = sigma * factor
    pub distance_factor: f32,

    pub velocity: RangeInclusive<f32>,
}

impl ParticleLattice {
    pub fn hex_square(&self, frame: &mut Frame, center: Vec2, particle_t: usize) {
        let total_particle_count = self.particle_count.0 * self.particle_count.1;
        if total_particle_count == 0 {
            return;
        }

        let meta = *frame.metadata();

        // Particle Radius x, y
        let rx = meta.particles[particle_t].force0_r() * self.distance_factor as f64;
        let ry = f64::sin(f64::consts::PI / 3.) * rx;

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

    pub fn square(&self, frame: &mut Frame, center: Vec2, particle_t: usize) {
        let total_particle_count = self.particle_count.0 * self.particle_count.1;
        if total_particle_count == 0 {
            return;
        }

        let meta = *frame.metadata();

        //let center = meta.box_size() / 2.;
        let r = meta.particles[particle_t].force0_r() * self.distance_factor as f64;
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

    pub fn random_vel(&self, rng: &mut ThreadRng) -> Vec2 {
        let v = self.velocity.clone().sample_single(rng).unwrap_or(0.);
        let angle = rng.random_range(0.0..2.0 * f32::consts::PI);
        let dir = Vec2::from(angle.sin_cos());
        dir * v as f64
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Preset {
    pub name: String,
    particles: [MiePotentialParams; 2],
    particles_list: Vec<Particle>,
}

pub struct Presets {
    path: Box<Path>,
    presets: Vec<Preset>,
}

impl Presets {
    pub fn new() -> Presets {
        Presets {
            path: Box::from(Path::new("presets")),
            presets: vec![Preset {
                name: "aaa".to_string(),
                particles: [
                    MiePotentialParams {
                        // Nitrogen
                        sigma: 3.609e-10,
                        epsilon: 105.79 * 1.,
                        n: 14.08,
                        m: 6.,
                    },
                    MiePotentialParams {
                        // Argon
                        sigma: 3.404e-10,
                        epsilon: 117.84 * 1.,
                        n: 12.085,
                        m: 6.,
                    },
                ],
                particles_list: vec![],
            }],
        }
    }

    pub fn readFromDisk(&self) {
        /*for entry in fs::read_dir(self.path.clone()).unwrap() {
            let path = entry.unwrap().path();
            let mut file = fs::File::open(path);
            let bytes;
            file.unwrap().read_to_end(bytes);
            let inp = from_bytes(bytes);

            //self.presets.push(preset);
        }*/
    }

    pub fn saveToDisk(&self) {}

    pub fn getPresetsLen(&self) -> usize {
        self.presets.len()
    }

    pub fn getPreset(&self, ind: usize) -> Preset {
        self.presets[ind].clone()
    }

    pub fn deletePreset(&self, preset: Frame) {}
}

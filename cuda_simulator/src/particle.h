#pragma once
#include <particle_io.h>

struct ParticleParams : MiePotentialParams {
    float C;
    double Cd;
    float mass = 6.63352599e-26;

    __host__ __device__ ParticleParams(MiePotentialParams p)
        : MiePotentialParams(p),
          C((n / (n - m)) * powf(n / m, n / (n - m))),
          Cd(((double)n / ((double)n - m)) * pow((double)n / m, (double)n / ((double)n - m))) {}

    __host__ __device__ double force_double(double r) const {
        double sr = sigma / r;
        return Cd * (double)epsilon * (m * pow(sr, m) - (double)n * pow(sr, (double)n)) / r;
    }

    __host__ __device__ float force(float r) const {
        float sr = sigma / r;
        return C * epsilon * (m * powf(sr, m) - n * powf(sr, n)) / r;
    }

    __host__ __device__ float2 force_double(Particle a, Particle b) const {
        double rx = (double)b.x - (double)a.x;
        double ry = (double)b.y - (double)a.y;
        double r = hypot(rx, ry);
        double f = force_double(r);

        f /= r;  // Normalize vector r
        return make_float2(f * rx, f * ry);
    }

    __host__ __device__ float2 force(Particle a, Particle b) const {
        float rx = b.x - a.x;
        float ry = b.y - a.y;
        float r = hypotf(rx, ry);
        float f = force(r);

        f /= r;  // Normalize vector r
        return make_float2(f * rx, f * ry);
    }

    __host__ __device__ void apply_force(Particle& dst, Particle src, float2 force,
                                         float dt) const {
        float ax = force.x / mass;
        float ay = force.y / mass;

        // x(dt) = x(0) + v(dt/2) * dt
        dst.x = src.x + src.vx * dt;
        dst.y = src.y + src.vy * dt;

        // v(dt+dt/2) = v(dt/2) + a(dt) * dt
        dst.vx = src.vx + ax * dt;
        dst.vy = src.vy + ay * dt;

        dst.ty = src.ty;
    }
};

__host__ __device__ void bounce_walls(Particle& p, float2 box_size) {
    if (p.x < 0. || p.x >= box_size.x) p.vx = -p.vx;
    if (p.y < 0. || p.y >= box_size.y) p.vy = -p.vy;
}

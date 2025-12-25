#pragma once
#include <particle_io.h>
#include <cassert>
#include <cstdint>

constexpr float k_b = 1.380649e-23;

// # Desmos
// C\ =\ \frac{n}{n-m}\left(\frac{n}{m}\right)^{\frac{m}{n-m}}
// F\left(r\right)
// =C\cdot p\cdot\frac{m\left(\frac{s}{r}\right)^{m}-n\left(\frac{s}{r}\right)^{n}}{r}
// V\left(r\right) =C\cdot p\left(\left(\frac{s}{r}\right)^{n}-\left(\frac{s}{r}\right)^{m}\right)
//
// Force Zero at:
// x=s\sqrt[n-m]{\frac{n}{m}}
//
// Max Attraction force at:
// x=s\sqrt[n-m]{\frac{\left(n+1\right)n}{\left(m+1\right)m}}
//
// # Wolframalpha
// Partial[\(40)Power[\(40)Divide[s,x]\(41),n]-Power[\(40)Divide[s,x]\(41),m]\(41) ,x]

__device__ __host__ inline float2 operator+(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

__device__ __host__ inline float2& operator+=(float2& a, const float2& b) {
    a.x += b.x;
    a.y += b.y;
    return a;
}

__host__ __device__ double2 d_dist(Particle a, Particle b, const FrameMetadata& frame) {
    double dx = a.x < b.x ? double(b.x - a.x) : -double(a.x - b.x);
    double dy = a.y < b.y ? double(b.y - a.y) : -double(a.y - b.y);

    const double max = (double)UINT32_MAX;
    return {(dx / max) * frame.box_width, (dy / max) * frame.box_height};
}

__host__ __device__ float2 f_dist(Particle a, Particle b, const FrameMetadata& frame) {
    float dx = a.x < b.x ? float(b.x - a.x) : -float(a.x - b.x);
    float dy = a.y < b.y ? float(b.y - a.y) : -float(a.y - b.y);

    const float max = (float)UINT32_MAX;
    return {(dx / max) * frame.box_width, (dy / max) * frame.box_height};
}

struct ParticleParams : MiePotentialParams {
    float C;
    float mass = 6.63352599e-26;

    __host__ __device__ ParticleParams(MiePotentialParams p) : MiePotentialParams(p) {
        C = (n / (n - m)) * powf(n / m, m / (n - m));
    }

    __host__ __device__ float d_force(double r) const {
        double dm = m, dn = n;
        double sr = (double)sigma / r;
        return C * epsilon * float(dm * pow(sr, dm) - dn * pow(sr, dn)) / (float)r;
    }

    __host__ __device__ float f_force(float r) const {
        float sr = sigma / r;
        return C * epsilon * (m * powf(sr, m) - n * powf(sr, n)) / r;
    }

    __host__ __device__ float f_force_repulsive(float r) const {
        float sr = sigma / r;
        return C * epsilon * m * powf(sr, m) / r;
    }

    __host__ __device__ float f_ljforce(float r) const {
        // assert(n == 12 && m == 6 && C == 4);
        float f2 = (sigma * sigma) / (r * r);
        float f6 = f2 * f2 * f2;
        float f12 = f6 * f6;
        return 24.0 * epsilon * (f6 - 2.0 * f12) / r;
    }

    __host__ __device__ double d_ljforce(double r) const {
        // assert(n == 12 && m == 6 && C == 4);
        double f2 = ((double)sigma * (double)sigma) / (r * r);
        double f6 = f2 * f2 * f2;
        double f12 = f6 * f6;
        return 24.0 * epsilon * (f6 - 2.0 * f12) / r;
    }

    __host__ __device__ float2 d2_force(double2 r) const {
        double len = hypot(r.x, r.y);
        float f = d_force(len);

        f /= len;  // Normalize vector r
        return make_float2(f * r.x, f * r.y);
    }

    __host__ __device__ float2 f2_force(float2 r) const {
        float len = hypotf(r.x, r.y);
        float f = f_force(len);

        f /= len;  // Normalize vector r
        return make_float2(f * r.x, f * r.y);
    }

    __host__ __device__ void f_apply_force(Particle& dst, Particle src, float2 force,
                                           const FrameMetadata& frame) const {
        float ax = force.x / mass;
        float ay = force.y / mass;

        // v(dt/2) = v(-dt/2) + a(0) * dt
        dst.vx = src.vx + ax * frame.step_dt;
        dst.vy = src.vy + ay * frame.step_dt;

        // x(dt) = x(0) + v(dt/2) * dt
        float dx = dst.vx * frame.step_dt;
        float dy = dst.vy * frame.step_dt;

        float u32_max = (float)UINT32_MAX;
        dst.x = src.x + (uint32_t)(int64_t)roundf((dx / frame.box_width) * u32_max);
        dst.y = src.y + (uint32_t)(int64_t)roundf((dy / frame.box_height) * u32_max);

        dst.ty = src.ty;
    }

    __host__ __device__ float2 f_wall_force(Particle p, const FrameMetadata& frame) const {
        const float max = (float)UINT32_MAX;

        float2 force;
        if (p.x < UINT32_MAX / 2) {
            float wall_left = (p.x / max) * frame.box_width;
            force.x = f_force_repulsive(wall_left);
        } else {
            float wall_right = ((UINT32_MAX - p.x) / max) * frame.box_width;
            force.x = -f_force_repulsive(wall_right);
        }
        if (p.y < UINT32_MAX / 2) {
            float wall_bottom = (p.y / max) * frame.box_height;
            force.y = f_force_repulsive(wall_bottom);
        } else {
            float wall_top = ((UINT32_MAX - p.y) / max) * frame.box_height;
            force.y = -f_force_repulsive(wall_top);
        }
        return force;
    }

    float f_force0_r() {
        return sigma * powf(n / m, 1.f / (n - m));
    }

    double d_force0_r() {
        double dn = n, dm = m, dsigma = sigma;
        return dsigma * pow(dn / dm, 1. / (dn - dm));
    }

    float f_max_attraction_r() {
        float ratio = ((n + 1.f) * n) / ((m + 1.f) * m);
        return sigma * powf(ratio, 1.f / (n - m));
    }

    double d_max_attraction_r() {
        double dn = n, dm = m, dsigma = sigma;
        double ratio = ((n + 1.) * n) / ((m + 1.) * m);
        return dsigma * pow(ratio, 1. / (dn - dm));
    }
};

__host__ __device__ void bounce_walls(Particle& p, float2 box_size) {
    if (p.x < 0. || p.x >= box_size.x) p.vx = -p.vx;
    if (p.y < 0. || p.y >= box_size.y) p.vy = -p.vy;
}

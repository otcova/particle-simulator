const rtx_extra_radius_scale = 1.2;
const u32_max = f32(0xFFFFFFFFu);
const antialias = 1.5;

struct Particle {
    @location(0) pos: vec2<u32>,
    @location(1) vel: vec2<f32>,
    @location(2) ty: i32,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) tex_coord: vec2<f32>, // -1 .. 1
    @location(2) tex_pixel_size: f32,
    @location(3) instance_index: u32,
}

struct MiePotentialParams {
    // Distance (meters) at which V = 0
    sigma: f32,
    // Dispersion energy (J)
    epsilon: f32,
    n: f32,
    m: f32,
}

struct FrameMetadata {
    particles: array<MiePotentialParams, 2>,
    step_dt: f32,
    steps_per_frame: u32,
    box_width: f32,
    box_height: f32,
    data_structure: u32,
    _padding1: u32,
    _padding2: u32,
    _padding3: u32,
}

struct Uniform {
    metadata: FrameMetadata,
    subtract_color: vec3<f32>,
    rtx: u32,
    real_time: f32,
    frame_time: f32,
    sim_time: f32,
    max_speed: f32,
    pixel_size: f32,
    min_particle_size: f32,
}

@group(0) @binding(0)
var<uniform> udata: Uniform;

const quad_verticies = array<vec2f, 4>(
    vec2(0., 0.) - 0.5,
    vec2(0., 1.) - 0.5,
    vec2(1., 0.) - 0.5,
    vec2(1., 1.) - 0.5,
);

@vertex
fn vertex_shader(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
    particle: Particle,
) -> VertexOutput {
    var out: VertexOutput;

    if particle.ty < 0 {
        out.position = vec4f(0., 0., 1., 0.);
        return out;
    }

    let box_size = vec2(udata.metadata.box_width, udata.metadata.box_height);
    var particle_size = udata.metadata.particles[0].sigma;

    if particle.ty < 2 {
        particle_size = udata.metadata.particles[particle.ty].sigma;
    }

    // Force a minimum particle size
    let min_size = udata.pixel_size * udata.min_particle_size;
    particle_size = max(particle_size, min_size) + udata.pixel_size * antialias;

    if udata.rtx == 2u {
        particle_size *= rtx_extra_radius_scale;
    }

    let quad_vertex = quad_verticies[vertex_index];
    let relative_speed = log2(1. + length(particle.vel)) / log2(1. + udata.max_speed);
    let p = (vec2f(particle.pos) / u32_max) * box_size;
    var pos = p + particle.vel * (udata.sim_time - udata.frame_time);
    let vertex = quad_vertex * particle_size;

    out.position = vec4f((pos + vertex) * 2. / box_size - 1., 0., 1.);
    out.color = mix(vec3f(0.0, 0.2, 1.), vec3f(1., 0.2, 0.0), relative_speed);
    out.tex_coord = quad_vertex * 2.;
    out.tex_pixel_size = udata.pixel_size * 2. / particle_size;
    out.instance_index = instance_index;

    return out;
}

@fragment
fn fragment_shader(in: VertexOutput) -> @location(0) vec4f {
    let r = length(in.tex_coord);

    var color: vec4f;

    if udata.rtx == 1u {
        color = shiny_circle(in, 1.);
    } else if udata.rtx == 2u {
        color = shiny2_circle(in);
    } else {
        // antialias
        let opacity = 1. - smoothstep(1. - in.tex_pixel_size * antialias, 1., r);
        color = vec4(in.color, opacity);
    }

    color = vec4(color.rgb - udata.subtract_color, color.a);
    return color;
}

const pi = 3.1415926535;
const tau = 3.1415926535 * 2.0;//radians(360.0);

fn shiny2_circle(in: VertexOutput) -> vec4f {
    let full_r = length(in.tex_coord);
    let r = full_r * rtx_extra_radius_scale;
    let a = atan2(in.tex_coord.y, in.tex_coord.x) / tau;

    let salt = in.instance_index;
    let t = 2. * udata.real_time + f32(salt);

    // Get the color
    var xCol = (a + ((100. + t) / 3.0)) * 3.0;
    xCol = xCol % 3.0;
    var horColour = vec3(0.25, 0.25, 0.25);

    if xCol < 1.0 {

        horColour.r += 1.0 - xCol;
        horColour.g += xCol;
    } else if xCol < 2.0 {

        xCol -= 1.0;
        horColour.g += 1.0 - xCol;
        horColour.b += xCol;
    } else {

        xCol -= 2.0;
        horColour.b += 1.0 - xCol;
        horColour.r += xCol;
    }

    // draw color beam
    let d = (r - 1.) / (1. + in.tex_pixel_size * 5.);
    let beamWidth = (2.7 + 0.5 * cos(a * 5.0 * tau)) * abs(1.0 / (30.0 * d));
    var horBeam = vec3(beamWidth);

    var opacity = min(1., beamWidth);
    var color = horBeam * horColour;

    if d < 0. {
        let c = shiny_circle(in, 1.1).rgb;
        color = mix(c, color, opacity / (abs(d) * 2. + 1.));
        opacity = 1.;
    }

    let transparency_start = 1. / rtx_extra_radius_scale - in.tex_pixel_size * antialias;
    opacity *= 1. - smoothstep(transparency_start, 1., full_r);
    return vec4(color, opacity);
}

fn shiny_circle(in: VertexOutput, size: f32) -> vec4f {
    let r = length(in.tex_coord);

    let x = in.tex_coord.x;
    let y = in.tex_coord.y;
    let z = sqrt(r * r - x * x - y * y);

    var color = in.color;

    let shade2 = smoothstep(size * 0.4, size, r);
    color *= 1. - shade2 * 0.3;

    let shade3 = smoothstep(size * 0.1, size, r);
    color *= 1. - shade2 * 0.2;


    let specular_pos = vec2(-0.1, 0.1) * size;
    let specular = smoothstep(0.6 * size, -0.2 * size, length(in.tex_coord - specular_pos));
    color += specular * 0.2;

    let opacity = 1. - smoothstep(size - in.tex_pixel_size * antialias, size, r);
    return vec4f(color, opacity);
}

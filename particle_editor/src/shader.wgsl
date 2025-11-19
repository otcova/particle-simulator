const radius_margin = 0.3;

struct Particle {
    @location(0) pos: vec2f,
    @location(1) vel: vec2f,
    @location(2) ty: u32,
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) color: vec3f,
    @location(1) tex_coord: vec2f,
    @location(2) instance_index: u32,
}

struct Uniform {
    rtx: u32,
    time: f32,
    radius: f32,
}

@group(0) @binding(0)
var<uniform> uniform_data: Uniform;

@vertex
fn vertex_shader(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
    particle: Particle,
) -> VertexOutput {
    var out: VertexOutput;

    if particle.ty == 0 {
        out.position = vec4f(0., 0., 1., 0.);
        return out;
    }

    // The incenter radius or equilateral triangle is sqrt(3)/6
    let edge_len = uniform_data.radius * ((1. + radius_margin) * 6. / sqrt(3.));
    let vertex_pos = array<vec2f, 3>(
        vec2(0.0, edge_len * 2. / 3.),
        vec2(-edge_len / 2., -edge_len / 3.),
        vec2(edge_len / 2., -edge_len / 3.)
    );

    let pos = particle.pos;
    let tex_coord = vertex_pos[vertex_index];
    let relative_speed = 0.3;

    let z = 0.; // select(0., -1., particle_is_nan(particle));
    out.position = vec4f((pos * 2. - 1.) + tex_coord, z, 1.);
    out.color = mix(vec3f(0.0, 0.2, 1.), vec3f(1., 0.2, 0.0), relative_speed);
    out.tex_coord = tex_coord;
    out.instance_index = instance_index;
    return out;
}

@fragment
fn fragment_shader(in: VertexOutput) -> @location(0) vec4f {
    let r = length(in.tex_coord / uniform_data.radius);
    if r > radius_margin + 1. {
        discard;
    }

    var color: vec4f;

    if uniform_data.rtx == 1u {
        return shiny_circle(in.tex_coord / uniform_data.radius, in.color);
    } else if uniform_data.rtx == 2u {
        return shiny2_circle(in.tex_coord / uniform_data.radius, in.color, in.instance_index);
    } else {
        let opacity = 1. - smoothstep(0.95, 1., r);
        return vec4(in.color, opacity);
    }
}

const pi = 3.1415926535;
const tau = 3.1415926535 * 2.0;//radians(360.0);

fn shiny2_circle(tex_coord: vec2<f32>, base_color: vec3<f32>, salt: u32) -> vec4f {
    let r = length(tex_coord);
    let a = atan2(tex_coord.y, tex_coord.x) / tau;

    let t = uniform_data.time + f32(salt);

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
    let d = r - 0.9;
    let beamWidth = (2.7 + 0.5 * cos(a * 5.0 * tau)) * abs(1.0 / (30.0 * d));
    var horBeam = vec3(beamWidth);

    var opacity = beamWidth;
    var color = horBeam * horColour;

    if d < 0. {
        let c = shiny_circle(tex_coord / 0.9, base_color).rgb;
        color = mix(c, color, opacity);
        opacity = 1.;
    }

    opacity = mix(opacity, 0., smoothstep(0.99, 1. + radius_margin, r));
    return vec4(color, opacity);
}

fn shiny_circle(tex_coord: vec2<f32>, base_color: vec3<f32>) -> vec4f {
    let r = length(tex_coord);

    let x = tex_coord.x;
    let y = tex_coord.y;
    let z = sqrt(r * r - x * x - y * y);

    var color = base_color;

    let shade1 = smoothstep(0.85, 1., r);
    color *= 1. - shade1 * 0.8;

    let shade2 = smoothstep(0.4, 1., r);
    color *= 1. - shade2 * 0.3;

    let shade3 = smoothstep(0.1, 1., r);
    color *= 1. - shade2 * 0.2;


    let specular_pos = vec2(-0.1, 0.1);
    let specular = smoothstep(0.6, -0.2, length(tex_coord - specular_pos));
    color += specular * 0.4;

    let opacity = 1. - smoothstep(0.95, 1., r);
    return vec4f(color, opacity);
}

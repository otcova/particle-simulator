override radius: f32 = 1.;

struct Particle {
    @location(0) pos: vec2f,
    @location(1) vel: vec2f,
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) color: vec3f,
    @location(1) tex_coord: vec2f,
}

struct Uniform {
    rtx: u32,
}

@group(0) @binding(0)
var<uniform> uniform_data: Uniform;

@vertex
fn vertex_shader(
    @builtin(vertex_index) vertex_index: u32,
    particle: Particle,
) -> VertexOutput {
    // The incenter radius or equilateral triangle is sqrt(3)/6
    let edge_len = radius * 6. / sqrt(3.);
    let vertex_pos = array<vec2f, 3>(
        vec2(0.0, edge_len * 2. / 3.),
        vec2(-edge_len / 2., -edge_len / 3.),
        vec2(edge_len / 2., -edge_len / 3.)
    );

    let pos = particle.pos;
    let tex_coord = vertex_pos[vertex_index];
    let relative_speed = 0.3;

    var out: VertexOutput;
    let z = 0.; // select(0., -1., particle_is_nan(particle));
    out.position = vec4f((pos * 2. - 1.) + tex_coord, z, 1.);
    out.color = mix(vec3f(0.0, 0.2, 1.), vec3f(1., 0.2, 0.0), relative_speed);
    out.tex_coord = tex_coord;
    return out;
}

@fragment
fn fragment_shader(in: VertexOutput) -> @location(0) vec4f {
    let d = length(in.tex_coord / radius);
    if d > 1. {
        discard;
    }

    var color = in.color;
    if uniform_data.rtx != 0u {
        color = shiny_circle(in.tex_coord / radius, in.color);
    }

    // Antialias
    let opacity = 1. - smoothstep(0.95, 1., d);
    return vec4(color, opacity);
}


fn shiny_circle(tex_coord: vec2<f32>, base_color: vec3<f32>) -> vec3f {
    let d = length(tex_coord);

    let x = tex_coord.x;
    let y = tex_coord.y;
    let z = sqrt(d * d - x * x - y * y);

    var color = base_color;

    let shade1 = smoothstep(0.85, 1., d);
    color *= 1. - shade1 * 0.5;

    let shade2 = smoothstep(0.4, 1., d);
    color *= 1. - shade2 * 0.3;

    let shade3 = smoothstep(0.1, 1., d);
    color *= 1. - shade2 * 0.2;


    let specular_pos = vec2(-0.1, 0.1);
    let specular = smoothstep(0.6, -0.2, length(tex_coord - specular_pos));
    color += specular * 0.4;

    return color;
}

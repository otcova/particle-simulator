use crate::wgpu_utils::WgpuContext;
use bytemuck::{Pod, Zeroable, cast_slice_mut};
use particle_io::{Frame, FrameMetadata};
use std::time::Instant;
use std::{mem::offset_of, num::NonZero};
use wgpu::{BindGroupLayoutEntry, hal::Rect};

#[repr(C)]
#[derive(Clone, Copy, Default, Zeroable, Pod)]
pub struct Uniform {
    metadata: FrameMetadata,
    pub rtx: u32,
    real_time: f32,
    pub frame_time: f32,
    pub simulation_time: f32,
    pub max_speed: f32,
    pixel_size: f32,
    pub min_particle_size: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Zeroable, Pod)]
pub struct WgpuParticle {
    pos: [u32; 2],
    vel: [f32; 2],
    ty: i32,
}

pub struct Graphics {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    pub background_color: [u8; 3],

    particles_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    pub uniform: Uniform,

    start_instant: Instant,
}

impl Graphics {
    pub fn new(gpu: &WgpuContext) -> Graphics {
        let bind_group_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });

        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let shader = gpu
            .device
            .create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Render Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vertex_shader"),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<WgpuParticle>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &[
                            wgpu::VertexAttribute {
                                shader_location: 0,
                                offset: offset_of!(WgpuParticle, pos) as u64,
                                format: wgpu::VertexFormat::Uint32x2,
                            },
                            wgpu::VertexAttribute {
                                shader_location: 1,
                                offset: offset_of!(WgpuParticle, vel) as u64,
                                format: wgpu::VertexFormat::Float32x2,
                            },
                            wgpu::VertexAttribute {
                                shader_location: 2,
                                offset: offset_of!(WgpuParticle, ty) as u64,
                                format: wgpu::VertexFormat::Sint32,
                            },
                        ],
                    }],
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &[],
                        zero_initialize_workgroup_memory: false,
                    },
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fragment_shader"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: gpu.surface_format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        // blend: Some(wgpu::BlendState {
                        //     color: wgpu::BlendComponent {
                        //         src_factor: wgpu::BlendFactor::One,
                        //         dst_factor: wgpu::BlendFactor::One,
                        //         operation: wgpu::BlendOperation::Add,
                        //     },
                        //     alpha: wgpu::BlendComponent::OVER,
                        // }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &[],
                        zero_initialize_workgroup_memory: false,
                    },
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None, //Some(wgpu::Face::Back),
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
                cache: None,
            });

        let particles_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Particles buffer"),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            size: 0,
            mapped_at_creation: false,
        });

        let uniform_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Uniform buffer"),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            size: size_of::<Uniform>() as wgpu::BufferAddress,
            mapped_at_creation: false,
        });

        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
            label: None,
        });

        let uniform = Uniform {
            rtx: 1,
            max_speed: 5000.,
            min_particle_size: 5.,
            ..Default::default()
        };

        Self {
            pipeline,
            bind_group,
            background_color: [5, 20, 40],

            particles_buffer,
            uniform_buffer,
            uniform,

            start_instant: Instant::now(),
        }
    }

    pub fn canvas_size(
        &mut self,
        metadata: &FrameMetadata,
        mut rect: wgpu::hal::Rect<u32>,
    ) -> wgpu::hal::Rect<u32> {
        let ratio = metadata.box_width / metadata.box_height; // w / h
        if rect.w as f32 > rect.h as f32 * ratio {
            let new_w = (rect.h as f32 * ratio) as u32;
            rect.x += (rect.w - new_w) / 2;
            rect.w = new_w;
        } else {
            let new_h = (rect.w as f32 / ratio) as u32;
            rect.y += (rect.h - new_h) / 2;
            rect.h = new_h;
        }

        rect
    }

    pub fn render(
        &mut self,
        gpu: &WgpuContext,
        encoder: &mut wgpu::CommandEncoder,
        frame: &Frame,
        rect: Rect<u32>,
    ) {
        self.update_particles(gpu, frame);
        self.update_uniform(gpu, frame, rect.clone());

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: gpu.surface_view.as_ref().unwrap(),
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: (self.background_color[0] as f64) / 255.,
                        g: (self.background_color[1] as f64) / 255.,
                        b: (self.background_color[2] as f64) / 255.,
                        a: 1.,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        if !frame.particles().is_empty() {
            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_viewport(
                rect.x as f32,
                rect.y as f32,
                rect.w as f32,
                rect.h as f32,
                0.,
                1.,
            );
            // render_pass.set_scissor_rect(rect.x, rect.y, rect.w, rect.h);
            let data_size = size_of::<WgpuParticle>() * frame.particles().len();
            render_pass.set_vertex_buffer(0, self.particles_buffer.slice(0..data_size as u64));
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.draw(0..4, 0..frame.particles().len() as u32);
        }
    }

    fn update_uniform(&mut self, gpu: &WgpuContext, frame: &Frame, rect: Rect<u32>) {
        self.uniform.pixel_size = frame.metadata().box_width / rect.w as f32;
        self.uniform.real_time = self.start_instant.elapsed().as_secs_f32();
        self.uniform.metadata = *frame.metadata();

        let bytes = bytemuck::bytes_of(&self.uniform);

        gpu.queue.write_buffer(&self.uniform_buffer, 0, bytes);
    }

    fn update_particles(&mut self, gpu: &WgpuContext, frame: &Frame) {
        let data_size = size_of::<WgpuParticle>() * frame.particles().len();

        let Some(data_size) = NonZero::new(data_size as u64) else {
            return;
        };

        if self.particles_buffer.size() < data_size.into() {
            self.particles_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Particles buffer"),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                size: data_size.into(),
                mapped_at_creation: false,
            });
        }

        let Some(mut dst_bytes) = gpu
            .queue
            .write_buffer_with(&self.particles_buffer, 0, data_size)
        else {
            return;
        };

        let dst: &mut [WgpuParticle] = cast_slice_mut(&mut dst_bytes);
        for (idx, particle) in frame.particles().iter().enumerate() {
            dst[idx].pos = particle.pos_u32();
            dst[idx].vel = particle.vel_f32();
            dst[idx].ty = particle.ty as i32;
        }
    }
}

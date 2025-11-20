use std::time::Instant;

use crate::wgpu_utils::WgpuContext;
use particle_io::{Frame, Particle};
use wgpu::{BindGroupLayoutEntry, hal::Rect, util::DeviceExt};

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Zeroable, bytemuck::NoUninit)]
pub struct Uniform {
    pub rtx: u32,
    time: f32,
    radius: f32,
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
                        array_stride: std::mem::size_of::<Particle>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &[
                            wgpu::VertexAttribute {
                                shader_location: 0,
                                offset: 0,
                                format: wgpu::VertexFormat::Float32x2,
                            },
                            wgpu::VertexAttribute {
                                shader_location: 1,
                                offset: size_of::<[f32; 2]>() as wgpu::BufferAddress,
                                format: wgpu::VertexFormat::Float32x2,
                            },
                            wgpu::VertexAttribute {
                                shader_location: 2,
                                offset: size_of::<[f32; 4]>() as wgpu::BufferAddress,
                                format: wgpu::VertexFormat::Uint32,
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
            radius: 0.05,
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

    pub fn canvas_size(&mut self, mut rect: Rect<u32>) -> Rect<u32> {
        let ratio = 1.; // w / h
        if rect.w > rect.h {
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
        self.update_uniform(gpu);

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
            render_pass.set_vertex_buffer(0, self.particles_buffer.slice(..));
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.draw(0..3, 0..frame.particles().len() as u32);
        }
    }

    fn update_uniform(&mut self, gpu: &WgpuContext) {
        self.uniform.time = self.start_instant.elapsed().as_secs_f32();

        let bytes = bytemuck::bytes_of(&self.uniform);
        gpu.queue.write_buffer(&self.uniform_buffer, 0, bytes);
    }

    fn update_particles(&mut self, gpu: &WgpuContext, frame: &Frame) {
        let data = bytemuck::cast_slice(frame.particles());

        if data.is_empty() {
            return;
        }

        if self.particles_buffer.size() != data.len() as wgpu::BufferAddress {
            self.particles_buffer =
                gpu.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Particles buffer"),
                        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                        contents: data,
                    });
        } else {
            gpu.queue.write_buffer(&self.particles_buffer, 0, data);
        }
    }
}

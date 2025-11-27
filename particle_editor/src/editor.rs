use std::sync::Arc;

use egui::{
    CentralPanel, CollapsingHeader, Color32, ComboBox, DragValue, Grid, Key, KeyboardShortcut,
    Margin, Modifiers, Pos2, Rect, ScrollArea, SidePanel, Slider, Vec2, WidgetText,
};
use particle_io::{Frame, FrameMetadata};
use winit::{
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::ActiveEventLoop,
    window::{Fullscreen, Window},
};

use crate::{
    backend::Backend,
    egui_utils::{EguiContext, NumFormat, NumFormatter, rect_from_pixels},
    graphics::Graphics,
    simulation::{Simulation, TimelineFrame},
    wgpu_utils::WgpuContext,
};

pub struct Editor {
    gpu: WgpuContext,
    egui: EguiContext,
    graphics: Graphics,
    simulation: Simulation,
    backend: Backend,

    ui_scale: f32,
    box_size: u32,
    floating_windows: bool,
    close_window: bool,
    interpolation: Interpolation,

    num_formatter: NumFormatter,

    sim_params: FrameMetadata,

    // play related
    play_time: f32,
    play_speed: f32,
    auto_play: bool,
    loop_play: bool,
}

impl Editor {
    pub async fn new(window: Arc<Window>) -> Editor {
        let gpu = WgpuContext::new(window).await;

        let editor = Editor {
            egui: EguiContext::new(&gpu, None, 1),
            graphics: Graphics::new(&gpu),
            simulation: Simulation::new(),
            backend: Backend::new(),
            gpu,

            ui_scale: 1.15,
            box_size: 5,
            floating_windows: false,
            close_window: false,
            interpolation: Interpolation::Velocity,

            num_formatter: NumFormatter {
                figures: 3,
                format: NumFormat::Metric,
                rgb: [140, 140, 180],
            },

            sim_params: FrameMetadata {
                ..Default::default()
            },

            // play related
            play_time: 0.,
            play_speed: 1e-9,
            auto_play: false,
            loop_play: false,
        };

        editor.egui.context().style_mut(|style| {
            style.visuals.handle_shape = egui::style::HandleShape::Rect { aspect_ratio: 0.5 };
            style.spacing.scroll.bar_width = 6.;
            style.spacing.scroll.foreground_color = false;
        });

        egui_extras::install_image_loaders(editor.egui.context());
        editor
    }

    pub fn render(&mut self) {
        self.gpu.window.request_redraw();

        self.simulation.update(&mut self.backend);

        if self.auto_play {
            let dt = self.egui.context().input(|i| i.stable_dt);
            self.play_time += dt * self.play_speed;

            if self.play_time > self.simulation.sim_len() {
                if self.loop_play {
                    self.play_time = 0.;
                } else {
                    self.play_time = self.simulation.sim_len();
                }
            }
        }

        self.gpu.start_frame();
        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());

        self.egui.begin_frame(&self.gpu.window);
        self.gui(&mut encoder);
        self.egui.end_frame_and_draw(&self.gpu, &mut encoder);

        self.gpu.queue.submit([encoder.finish()]);
        self.gpu.end_frame();
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        self.gpu.resize(new_size);
    }

    pub fn handle_input(&mut self, event_loop: &ActiveEventLoop, event: &WindowEvent) {
        if self.close_window {
            event_loop.exit();
            return;
        }

        self.egui.handle_input(&self.gpu.window, event);
    }

    fn gui(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let ctx = &mut self.egui.context().clone();

        if self.floating_windows {
            CentralPanel::default()
                .frame(egui::Frame::NONE)
                .show(ctx, |ui| {
                    self.left_panel(ui);
                    self.playback_panel(ui);
                });
        } else {
            SidePanel::left("left").resizable(false).show(ctx, |ui| {
                ScrollArea::vertical().show(ui, |ui| {
                    egui::Frame::new()
                        .inner_margin(Margin {
                            left: 5,
                            right: 20,
                            top: 10,
                            bottom: 10,
                        })
                        .show(ui, |ui| self.left_panel(ui))
                });
            });

            egui::TopBottomPanel::bottom("bottom")
                .resizable(false)
                .show(ctx, |ui| {
                    egui::Frame::new()
                        .inner_margin(Margin {
                            left: 5,
                            right: 5,
                            top: 5,
                            bottom: 5,
                        })
                        .show(ui, |ui| self.playback_panel(ui));
                });
        }

        self.egui.set_ui_scale(self.ui_scale);

        CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ctx, |ui| self.draw_canvas(ui, encoder));

        ctx.input_mut(|i| self.keyboard_shortcuts(i));
    }

    fn draw_canvas(&mut self, ui: &egui::Ui, encoder: &mut wgpu::CommandEncoder) {
        let rect_points = ui.available_rect_before_wrap();

        let mut rect = Rect::from_min_max(
            rect_points.min * ui.pixels_per_point(),
            rect_points.max * ui.pixels_per_point(),
        );

        rect.max = rect.max.min(egui::Pos2 {
            x: (self.gpu.surface_size.width - 1) as f32,
            y: (self.gpu.surface_size.height - 1) as f32,
        });

        let TimelineFrame {
            frame, frame_time, ..
        } = self.simulation.frame(self.play_time);

        let canvas_rect = self.graphics.canvas_size(
            frame.metadata(),
            wgpu::hal::Rect {
                x: rect.min.x as u32,
                y: rect.min.y as u32,
                w: rect.size().x as u32,
                h: rect.size().y as u32,
            },
        );

        let outter = rect_points;
        let inner = rect_from_pixels(ui, canvas_rect.clone());

        match self.interpolation {
            Interpolation::None => {
                self.graphics.uniform.simulation_time = frame_time;
            }
            Interpolation::Velocity => {
                self.graphics.uniform.simulation_time = self.play_time;
            }
        }
        self.graphics.uniform.frame_time = frame_time;

        self.graphics.render(&self.gpu, encoder, frame, canvas_rect);

        let fill = Color32::from_black_alpha(100);
        ui.painter().rect_filled(
            Rect::from_min_max(outter.min, Pos2::new(inner.min.x, inner.max.y)),
            0,
            fill,
        );
        ui.painter().rect_filled(
            Rect::from_min_max(outter.min, Pos2::new(inner.max.x, inner.min.y)),
            0,
            fill,
        );
        ui.painter().rect_filled(
            Rect::from_min_max(Pos2::new(inner.min.x, inner.max.y), outter.max),
            0,
            fill,
        );
        ui.painter().rect_filled(
            Rect::from_min_max(Pos2::new(inner.max.x, inner.min.y), outter.max),
            0,
            fill,
        );
    }

    fn keyboard_shortcuts(&mut self, input: &mut egui::InputState) {
        let esc = KeyboardShortcut::new(Modifiers::NONE, Key::Escape);
        if input.consume_shortcut(&esc) {
            self.close_window = true;
        }

        let f11 = KeyboardShortcut::new(Modifiers::NONE, Key::F11);
        if input.consume_shortcut(&f11) {
            self.toggle_fullscreen();
        }

        let space = KeyboardShortcut::new(Modifiers::NONE, Key::Space);
        if input.consume_shortcut(&space) {
            self.auto_play = !self.auto_play;
        }

        let left = KeyboardShortcut::new(Modifiers::NONE, Key::ArrowLeft);
        if input.consume_shortcut(&left) {
            self.play_time = (self.play_time - self.play_speed).max(0.);
        }

        let right = KeyboardShortcut::new(Modifiers::NONE, Key::ArrowRight);
        if input.consume_shortcut(&right) {
            self.play_time = if self.play_time + self.play_speed > self.simulation.sim_len() {
                0.
            } else {
                self.play_time + self.play_speed
            };
        }

        // ignore this one
        let d = KeyboardShortcut::new(Modifiers::NONE, Key::D);
        if input.consume_shortcut(&d) {
            println!("{:?}", self.simulation.print(self.play_time));
        }
    }

    fn ui_section<F>(&mut self, ui: &mut egui::Ui, title: &'static str, content: F)
    where
        F: FnOnce(&mut Self, &mut egui::Ui),
    {
        if self.floating_windows {
            let ctx = &self.egui.context().clone();
            egui::Window::new(title).show(ctx, |ui| content(self, ui));
        } else {
            ui.heading(title);
            ui.separator();
            content(self, ui);
            ui.add_space(40.);
        }
    }

    fn toggle_fullscreen(&self) {
        let window = &self.gpu.window;
        if window.fullscreen().is_some() {
            window.set_fullscreen(None);
        } else {
            window.current_monitor().map(|monitor| {
                monitor.video_modes().next().map(|video_mode| {
                    window.set_fullscreen(Some(if cfg!(any(target_os = "macos", unix)) {
                        Fullscreen::Borderless(Some(monitor))
                    } else {
                        Fullscreen::Exclusive(video_mode)
                    }));
                })
            });
        }
    }

    fn left_panel(&mut self, ui: &mut egui::Ui) {
        self.ui_section(ui, "Backend", |editor, ui| {
            Grid::new("backend-grid").num_columns(2).show(ui, |ui| {
                if editor.backend.reader_connected() || editor.backend.writer_connected() {
                    let button = ui.button("Disconnect");
                    if button.clicked() {
                        editor.backend.close_connection();
                    }
                } else {
                    ui.label("Connect with ");
                    ui.horizontal(|ui| {
                        if ui.button("Files / Pipes").clicked() {
                            editor.backend.open_backend_files();
                        }
                    });
                }
                ui.end_row();

                ui.label("TCP Server ");
                ui.label(editor.backend.tcp_server_status());
                ui.end_row();
            });
            ui.add_space(5.);

            if !editor.backend.reader_details.is_empty() {
                ui.add_space(5.);
                ui.label(format!(
                    "{:?} Backend Output:\n  {}",
                    editor.backend.reader_state(),
                    editor.backend.reader_details
                ));
            }

            if !editor.backend.reader_details.is_empty() {
                ui.add_space(5.);
                ui.label(format!(
                    "{:?} Backend Input:\n  {}",
                    editor.backend.writer_state(),
                    editor.backend.writer_details
                ));
            }
        });

        self.ui_section(ui, "Editor", |editor, ui| {
            CollapsingHeader::new("Load Square")
                .default_open(true)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Size");
                        ui.add(Slider::new(&mut editor.box_size, 0..=20));
                    });

                    ui.add_enabled_ui(editor.backend.writer_connected(), |ui| {
                        if ui.button("Send To Backend").clicked() {
                            let mut frame = Frame::new();
                            *frame.metadata_mut() = editor.sim_params;
                            let pos = (
                                editor.sim_params.box_width / 2.,
                                editor.sim_params.box_height / 2.,
                            );
                            let size =
                                f32::min(editor.sim_params.box_width, editor.sim_params.box_height);
                            frame.push_square(pos, size, editor.box_size);
                            editor.backend.write(&frame);
                        }
                    });
                });
        });

        self.ui_section(ui, "Parameters", |editor, ui| {
            let mut params = editor.sim_params;

            Grid::new("params-grid").num_columns(2).show(ui, |ui| {
                ui.label("Step delta time");
                ui.add(
                    Slider::new(&mut params.step_dt, 1e-15..=1e-1)
                        .custom_formatter(|t, _| editor.num_formatter.raw_string(t as f32, "s"))
                        .logarithmic(true),
                );
                ui.end_row();

                ui.label("Steps per frame");
                ui.add(Slider::new(&mut params.steps_per_frame, 1..=1000000).logarithmic(true));
                ui.end_row();

                ui.label("Frame delta time");
                let frame_dt = params.step_dt * params.steps_per_frame as f32;
                ui.label(editor.num(frame_dt, "s"));
                ui.end_row();

                ui.end_row();

                let box_size_range = 1e-9..=1000e-9;
                ui.label("Box width");
                ui.add(
                    Slider::new(&mut params.box_width, box_size_range.clone())
                        .custom_formatter(|t, _| editor.num_formatter.raw_string(t as f32, "m"))
                        .logarithmic(true),
                );
                ui.end_row();

                ui.label("Box height");
                ui.add(
                    Slider::new(&mut params.box_height, box_size_range.clone())
                        .custom_formatter(|t, _| editor.num_formatter.raw_string(t as f32, "m"))
                        .logarithmic(true),
                );
                ui.end_row();

                ui.label("Data structure");

                let current_data_structure =
                    particle_io::DataStructure::try_from(params.data_structure)
                        .map(|d| d.name())
                        .unwrap_or("Invalid option");

                ComboBox::from_id_salt("Data structure")
                    .selected_text(current_data_structure)
                    .show_ui(ui, |ui| {
                        let d = &mut params.data_structure;
                        use particle_io::DataStructure::*;
                        ui.selectable_value(d, CompactArray as u32, CompactArray.name());
                        ui.selectable_value(d, MatrixBuckets as u32, MatrixBuckets.name());
                    });
                ui.end_row();
            });
            ui.add_space(20.);

            for (idx, particle) in params.particles.iter_mut().enumerate() {
                let name = format!("Particle {}", idx);
                ui.collapsing(&name, |ui| {
                    Grid::new(&name).num_columns(2).show(ui, |ui| {
                        ui.label("σ");
                        ui.add(
                            Slider::new(&mut particle.sigma, 1e-10..=10e-10)
                                .custom_formatter(|t, _| {
                                    editor.num_formatter.raw_string(t as f32, "m")
                                })
                                .logarithmic(true),
                        );
                        ui.end_row();

                        ui.label("Ɛ");
                        ui.add(Slider::new(&mut particle.epsilon, (1.)..=200.).logarithmic(true));
                        ui.end_row();

                        ui.label("n");
                        ui.add(Slider::new(&mut particle.n, (8.)..=16.));
                        ui.end_row();

                        ui.label("m");
                        ui.add(Slider::new(&mut particle.m, (3.)..=9.));
                        ui.end_row();
                    });
                });
                ui.end_row();
            }

            if editor.sim_params != params {
                // Maybe: Params changed => Send to backend
            }
            editor.sim_params = params;
        });

        self.ui_section(ui, "Stats", |editor, ui| {
            let TimelineFrame {
                frame,
                frame_time,
                frame_idx,
            } = editor.simulation.frame(editor.play_time);
            let particles_count = frame.particles().len() as f32;
            let frames_count = editor.simulation.frames_count() as f32;
            let total_sim_time = editor.simulation.sim_len();

            Grid::new("stats-grid").num_columns(2).show(ui, |ui| {
                ui.label("Time");
                ui.label(editor.num(editor.play_time, "s"));
                ui.end_row();

                ui.label("Frame Time");
                ui.horizontal(|ui| {
                    ui.label(editor.num(frame_time, "s"));
                    ui.label("/");
                    ui.label(editor.num_int(total_sim_time, "s"));
                });
                ui.end_row();

                ui.label("Frame Index");
                ui.horizontal(|ui| {
                    ui.label(editor.num_int((frame_idx + 1) as f32, ""));
                    ui.label("/");
                    ui.label(editor.num_int(frames_count, ""));
                });
                ui.end_row();

                ui.label("Step delta time");
                ui.label(editor.num(3.2113e-4, "s"));
                ui.end_row();

                ui.label("Num Particles");
                ui.label(editor.num(particles_count, ""));
                ui.end_row();

                ui.end_row();

                ui.label("Temperature");
                ui.vertical(|ui| {
                    ui.label(editor.num(100., "K"));
                    ui.label(editor.num(100. - 273.15, "ºC"));
                });
                ui.end_row();

                ui.label("2D Pressure");
                ui.label(editor.num(12345., "N/m"));
                ui.end_row();

                ui.end_row();

                ui.label("Kinetic energy");
                ui.label(editor.num(0.0000412412, "J"));
                ui.end_row();

                ui.label("Potential energy");
                ui.label(editor.num(0.000012122, "J"));
                ui.end_row();

                ui.label("Total energy");
                ui.label(editor.num(0.000051234124, "J"));
                ui.end_row();
            });
        });

        self.ui_section(ui, "Timeline", |editor, ui| {
            Grid::new("memory-grid").num_columns(2).show(ui, |ui| {
                ui.label("Timeline RAM");
                let ram = editor.simulation.timeline_ram();
                ui.label(editor.num_int(ram as f32, "B"));
                ui.end_row();
            });

            if ui.button("Clear Timeline").clicked() {
                editor.simulation.clear();
            }
        });

        self.ui_section(ui, "GUI", |editor, ui| {
            Grid::new("window-grid").num_columns(2).show(ui, |ui| {
                ui.label("GUI size");
                ui.add(
                    DragValue::new(&mut editor.ui_scale)
                        .range(0.5..=3.0)
                        .prefix("x")
                        .speed(0.01),
                );
                ui.end_row();

                ui.label("Number Format");
                ComboBox::from_id_salt("Number Format")
                    .selected_text(editor.num_formatter.format.name())
                    .show_ui(ui, |ui| {
                        let f = &mut editor.num_formatter.format;
                        use NumFormat::*;
                        ui.selectable_value(f, Dashed, Dashed.name());
                        ui.selectable_value(f, Scientific, Scientific.name());
                        ui.selectable_value(f, Metric, Metric.name());
                    });
                ui.end_row();

                ui.label("Significant Digits");
                ui.add(DragValue::new(&mut editor.num_formatter.figures).range(1..=8));
                ui.end_row();

                ui.label("Background Color");
                ui.color_edit_button_srgb(&mut editor.graphics.background_color);
                ui.end_row();

                ui.label("Number Color");
                ui.color_edit_button_srgb(&mut editor.num_formatter.rgb);
                ui.end_row();

                ui.end_row();

                ui.label("Max Speed for Color");
                ui.add(
                    Slider::new(&mut editor.graphics.uniform.max_speed, 1.0..=100_000.0)
                        .logarithmic(true)
                        .suffix("m/s"),
                );
                ui.end_row();

                ui.label("Min Particle Size");
                ui.add(
                    DragValue::new(&mut editor.graphics.uniform.min_particle_size)
                        .range(1.0..=40.0)
                        .speed(0.2)
                        .suffix("px"),
                );
                ui.end_row();

                let rtx_names = ["Off", "Ultra", "RGB"];
                let rtx = &mut editor.graphics.uniform.rtx;
                let rtx_idx = (*rtx as usize).min(rtx_names.len() - 1);

                ui.label("RTX");
                ComboBox::from_id_salt("RTX")
                    .selected_text(rtx_names[rtx_idx])
                    .show_ui(ui, |ui| {
                        ui.selectable_value(rtx, 0, rtx_names[0]);
                        ui.selectable_value(rtx, 1, rtx_names[1]);
                        ui.selectable_value(rtx, 2, rtx_names[2]);
                    });
                ui.end_row();

                ui.label("Interpolation");
                ComboBox::from_id_salt("Interpolation")
                    .selected_text(editor.interpolation.name())
                    .show_ui(ui, |ui| {
                        let v = &mut editor.interpolation;
                        use Interpolation::*;
                        ui.selectable_value(v, None, None.name());
                        ui.selectable_value(v, Velocity, Velocity.name());
                    });
                ui.end_row();
            });

            if ui.button("Full Screen").clicked() {
                editor.toggle_fullscreen();
            }

            if editor.floating_windows {
                if ui.button("Reconstruct").clicked() {
                    editor.floating_windows = false;
                }
            } else if ui.button("Boom").clicked() {
                editor.floating_windows = true;
            }
        });
    }

    fn num_int(&self, n: f32, unit: &'static str) -> WidgetText {
        let mut f = self.num_formatter;
        if n < 1000.0 {
            f.format = NumFormat::Metric;
            f.figures = 0;
            f.fmt(n, unit)
        } else {
            f.figures = u32::min(3, f.figures);
            f.fmt(n, unit)
        }
    }

    fn num(&self, n: f32, unit: &'static str) -> WidgetText {
        self.num_formatter.fmt(n, unit)
    }

    // can be optimized to only recalculate widgets width when necessary
    fn playback_panel(&mut self, ui: &mut egui::Ui) {
        let mut content = |ui: &mut egui::Ui| -> () {
            ui.vertical(|ui| {
                // timeline bar

                ui.horizontal(|ui| {
                    let t_time = self.simulation.sim_len();

                    let t_time_text = self.num_formatter.raw_string(t_time, "s");

                    let mut cursor = self.play_time;

                    ui.style_mut().spacing.slider_width = 0.;
                    let resp = ui.add_visible(
                        false,
                        egui::Slider::new(&mut cursor, (0.)..=t_time)
                            .suffix(format!(" /{t_time_text}"))
                            .custom_formatter(|n, _| self.num_formatter.raw_string(n as f32, "s")),
                    );

                    ui.add_space(-resp.rect.width() - 8.);
                    ui.style_mut().spacing.slider_width =
                        (0 as f32).max(ui.available_width() - resp.rect.width());

                    if t_time == 0. {
                        ui.add(
                            egui::Slider::new(&mut cursor, (0.)..=0.1)
                                .fixed_decimals(0)
                                .suffix(format!(" /{t_time_text}"))
                                .custom_formatter(|_, _| self.num_formatter.raw_string(0., "s")),
                        );
                    } else {
                        ui.add(
                            egui::Slider::new(&mut cursor, (0.)..=t_time)
                                .clamping(egui::SliderClamping::Never)
                                .suffix(format!(" /{t_time_text}"))
                                .trailing_fill(true)
                                .custom_formatter(|n, _| {
                                    self.num_formatter.raw_string(n as f32, "s")
                                }),
                        );
                    }

                    self.play_time = cursor;
                });

                // play buttons
                ui.horizontal(|ui| {
                    let tot_space = ui.available_width();
                    let l_ui_p = ui.max_rect().left();

                    // 22.96875: buttons width (pre-measured/observed)
                    let speed_space = tot_space / 2.
                        - (4. * (22.96875) / 2. + 3. * 8. / 2.) // entire button area width (except speed)
                        + (22.96875 / 2. + 8. / 2.); // to have the play button on center

                    const MIN_SPEED: f32 = 1e-15;
                    const MAX_SPEED: f32 = 1.0;

                    ui.style_mut().spacing.item_spacing =
                        Vec2::new(8.5 + ui.style().spacing.icon_width, 0.);
                    ui.add_space(
                        -(ui.style().spacing.slider_width + ui.style().spacing.icon_width + 8.),
                    );

                    ui.add(
                        egui::Slider::new(&mut self.play_speed, MIN_SPEED..=MAX_SPEED)
                            .custom_formatter(|n, _| self.num_formatter.raw_string(n as f32, "s/s"))
                            .logarithmic(true),
                    );

                    ui.style_mut().spacing.item_spacing = Vec2::new(8., 0.);

                    if ui
                        .put(
                            egui::Rect::from_min_max(
                                Pos2 {
                                    x: l_ui_p + speed_space,
                                    y: ui.cursor().top(),
                                },
                                Pos2 {
                                    x: l_ui_p + speed_space + 22.96875,
                                    y: ui.cursor().bottom(),
                                },
                            ),
                            egui::Button::image(egui::Image::new(egui::include_image!(
                                "../icons/media-seek-backward.png"
                            ))),
                        )
                        .clicked()
                    {
                        self.play_time = (self.play_time - self.play_speed).max(0.);
                    };

                    if ui
                        .add(egui::Button::image(egui::Image::new(if self.auto_play {
                            egui::include_image!("../icons/media-playback-pause.png")
                        } else {
                            egui::include_image!("../icons/media-playback-start.png")
                        })))
                        .clicked()
                    {
                        self.auto_play = !self.auto_play;
                    }

                    if ui
                        .add(egui::Button::image(egui::Image::new(egui::include_image!(
                            "../icons/media-seek-forward.png"
                        ))))
                        .clicked()
                    {
                        self.play_time =
                            if self.play_time + self.play_speed > self.simulation.sim_len() {
                                0.
                            } else {
                                self.play_time + self.play_speed
                            };
                    }

                    if ui
                        .add(egui::Button::image(
                            egui::Image::new(egui::include_image!(
                                "../icons/media-playlist-repeat.png"
                            ))
                            .tint(Color32::from_gray(if self.loop_play { 255 } else { 0 })),
                        ))
                        .clicked()
                    {
                        self.loop_play = !self.loop_play;
                    };
                });
            });
        };

        if self.floating_windows {
            let ctx = &self.egui.context().clone();
            egui::Window::new("Playback").show(ctx, content);
        } else {
            content(ui);
        }
    }
}

#[derive(PartialEq, Eq, Clone, Copy)]
enum Interpolation {
    None,
    Velocity,
}

impl Interpolation {
    pub fn name(self) -> &'static str {
        match self {
            Interpolation::None => "None",
            Interpolation::Velocity => "Velocity",
        }
    }
}

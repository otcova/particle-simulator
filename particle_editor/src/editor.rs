use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use egui::Color32;
use particle_io::Frame;
use wgpu::hal::Rect;
use winit::{
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::ActiveEventLoop,
    window::{Fullscreen, Window},
};

use crate::{
    backend::Backend, egui_utils::EguiContext, graphics::Graphics, simulation::Simulation,
    wgpu_utils::WgpuContext,
};

pub struct Editor {
    gpu: WgpuContext,
    egui: EguiContext,
    graphics: Graphics,
    simulation: Simulation,
    backend: Backend,

    ui_scale: f32,
    frame_index: u32,
    box_size: u32,
    floating_windows: bool,
    close_window: bool,

    // play related
    auto_play: bool,
    loop_play: bool,
    play_speed: f32,
    prev_instant: Instant,
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
            frame_index: 0,
            box_size: 5,
            floating_windows: false,
            close_window: false,

            // play related
            auto_play: false,
            loop_play: true,
            play_speed: 1.,
            prev_instant: Instant::now(),
        };
        egui_extras::install_image_loaders(editor.egui.context());
        editor
    }

    pub fn render(&mut self) {
        self.gpu.window.request_redraw();

        self.simulation.update(&mut self.backend);

        let cur_n_frames = self.simulation.timeline_frames_count();
        #[allow(clippy::collapsible_if)]
        if self.auto_play && cur_n_frames > 0 {
            if self.prev_instant.elapsed() > Duration::from_millis((1000. / self.play_speed) as u64)
            {
                self.prev_instant = Instant::now();

                self.frame_index += 1;
                if self.frame_index >= cur_n_frames {
                    if self.loop_play {
                        self.frame_index = 0;
                    } else {
                        self.frame_index = cur_n_frames - 1;
                    }
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
            egui::CentralPanel::default()
                .frame(egui::Frame::NONE)
                .show(ctx, |ui| {
                    self.left_panel(ui);
                    self.playback_panel(ui);
                });
        } else {
            egui::SidePanel::left("left")
                .resizable(false)
                .show(ctx, |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        egui::Frame::new()
                            .inner_margin(egui::Margin {
                                left: 5,
                                right: 15,
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
                        .inner_margin(egui::Margin {
                            left: 5,
                            right: 5,
                            top: 5,
                            bottom: 5,
                        })
                        .show(ui, |ui| self.playback_panel(ui));
                });
        }

        self.egui.set_ui_scale(self.ui_scale);
        let ppp = self.egui.context().pixels_per_point();

        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ctx, |ui| {
                let right_rect_points = ui.available_rect_before_wrap();
                let mut right_rect = egui::Rect::from_min_max(
                    right_rect_points.min * ppp,
                    right_rect_points.max * ppp,
                );

                right_rect.max = right_rect.max.min(egui::Pos2 {
                    x: (self.gpu.surface_size.width - 1) as f32,
                    y: (self.gpu.surface_size.height - 1) as f32,
                });

                self.graphics.render(
                    &self.gpu,
                    encoder,
                    self.simulation.frame(self.frame_index),
                    Rect {
                        x: right_rect.min.x as u32,
                        y: right_rect.min.y as u32,
                        w: right_rect.size().x as u32,
                        h: right_rect.size().y as u32,
                    },
                );
            });

        ctx.input_mut(|i| self.keyboard_shortcuts(i));
    }

    fn keyboard_shortcuts(&mut self, input: &mut egui::InputState) {
        let esc = egui::KeyboardShortcut::new(egui::Modifiers::NONE, egui::Key::Escape);
        if input.consume_shortcut(&esc) {
            self.close_window = true;
        }

        let f11 = egui::KeyboardShortcut::new(egui::Modifiers::NONE, egui::Key::F11);
        if input.consume_shortcut(&f11) {
            self.toggle_fullscreen();
        }

        let space = egui::KeyboardShortcut::new(egui::Modifiers::NONE, egui::Key::Space);
        if input.consume_shortcut(&space) {
            self.auto_play = !self.auto_play;
            self.prev_instant = Instant::now();
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
        self.ui_section(ui, "Window", |editor, ui| {
            if ui.button("Full Screen").clicked() {
                editor.toggle_fullscreen();
            }

            ui.horizontal(|ui| {
                ui.label("UI size ");
                ui.add(
                    egui::DragValue::new(&mut editor.ui_scale)
                        .range(0.5..=3.0)
                        .prefix("x")
                        .speed(0.01),
                );
            });

            if editor.floating_windows {
                if ui.button("Reconstruct").clicked() {
                    editor.floating_windows = false;
                }
            } else if ui.button("Boom").clicked() {
                editor.floating_windows = true;
            }
        });

        self.ui_section(ui, "Backend", |editor, ui| {
            ui.collapsing(
                format!("Backend Output: {:?}", editor.backend.reader_state()),
                |ui| {
                    ui.label(&editor.backend.reader_details);
                },
            );
            ui.collapsing(
                format!("Backend Input: {:?}", editor.backend.writer_state()),
                |ui| {
                    ui.label(&editor.backend.writer_details);
                },
            );
            ui.add_space(5.);

            if ui.button("Connect by files").clicked() {
                editor.backend.open_backend_files();
            }

            if ui.button("Connect by TCP").clicked() {
                editor.backend.open_tcp();
            }

            if editor.backend.reader_connected() || editor.backend.writer_connected() {
                let button = ui.button("Disconnect");
                if button.clicked() {
                    editor.backend.close_connection();
                }
            }
        });

        self.ui_section(ui, "Simulation", |editor, ui| {
            if ui.button("Clear Timeline").clicked() {
                editor.simulation.clear();
            }
        });

        self.ui_section(ui, "Editor", |editor, ui| {
            ui.collapsing("Load Square", |ui| {
                ui.horizontal(|ui| {
                    ui.label("Size ");
                    ui.add(egui::Slider::new(&mut editor.box_size, 0..=20));
                });

                ui.add_enabled_ui(editor.backend.writer_connected(), |ui| {
                    if ui.button("Send To Backend").clicked() {
                        let mut frame = Frame::new();
                        frame.push_square(editor.box_size);
                        editor.backend.write(&frame);
                    }
                });
            });
        });

        self.ui_section(ui, "Render", |editor, ui| {
            ui.horizontal(|ui| {
                ui.label("Background Color: ");
                ui.color_edit_button_srgb(&mut editor.graphics.background_color);
            });

            ui.horizontal(|ui| {
                let rtx_names = ["Off", "Ultra", "RGB"];
                let rtx = &mut editor.graphics.uniform.rtx;
                let rtx_idx = (*rtx as usize).min(rtx_names.len() - 1);

                ui.label("RTX: ");
                egui::ComboBox::from_id_salt("RTX: ")
                    .selected_text(rtx_names[rtx_idx])
                    .show_ui(ui, |ui| {
                        ui.selectable_value(rtx, 0, rtx_names[0]);
                        ui.selectable_value(rtx, 1, rtx_names[1]);
                        ui.selectable_value(rtx, 2, rtx_names[2]);
                    });
            });
        });
    }

    // can be optimized to only recalculate widgets width when necessary
    fn playback_panel(&mut self, ui: &mut egui::Ui) {
        let mut content = |ui: &mut egui::Ui| -> () {
            ui.vertical(|ui| {
                // timeline bar

                ui.horizontal(|ui| {
                    let frames_count = self.simulation.timeline_frames_count();
                    let mut cursor = if self.frame_index == 0 {
                        0
                    } else {
                        frames_count.min(self.frame_index + 1)
                    };

                    ui.style_mut().spacing.slider_width = 0.;
                    let resp = ui.add_visible(
                        false,
                        egui::Slider::new(&mut cursor, frames_count.min(1)..=frames_count)
                            .suffix(format!("/{}", frames_count))
                            .trailing_fill(true),
                    );

                    ui.add_space(-resp.rect.width() - 8.);
                    ui.style_mut().spacing.slider_width =
                        (0 as f32).max(ui.available_width() - resp.rect.width());

                    ui.add(
                        egui::Slider::new(&mut cursor, frames_count.min(1)..=frames_count)
                            .suffix(format!("/{}", frames_count))
                            .trailing_fill(true),
                    );

                    self.frame_index = cursor.saturating_sub(1);
                });

                // play buttons
                ui.horizontal(|ui| {
                    let tot_space = ui.available_width();
                    let resp =
                        ui.add_visible(false, egui::Button::new(self.play_speed.to_string() + "x"));

                    // 22.96875: buttons width (pre-measured/observed)
                    ui.add_space(
                        tot_space / 2.
                            - (4. * (22.96875) / 2. + 3. * 8. / 2.) // entire button area width (except speed)
                            + (22.96875 / 2. + 8. / 2.) // to have the play button on center
                            - (2. * resp.rect.width() + 2. * 8.), // to make the speed button/s not affect others positions
                    );

                    if ui.button(self.play_speed.to_string() + "x").clicked() {
                        const MAX_SPEED: f32 = 8.;
                        const MIN_SPEED: f32 = 0.5;

                        self.play_speed *= 2.;
                        if self.play_speed > MAX_SPEED {
                            self.play_speed = MIN_SPEED;
                        }
                    }
                    if ui
                        .add(egui::Button::image(egui::Image::new(egui::include_image!(
                            "../icons/media-seek-backward.png"
                        ))))
                        .clicked()
                    {
                        self.frame_index = self.frame_index.saturating_sub(1);
                    }

                    if ui
                        .add(egui::Button::image(egui::Image::new(if self.auto_play {
                            egui::include_image!("../icons/media-playback-pause.png")
                        } else {
                            egui::include_image!("../icons/media-playback-start.png")
                        })))
                        .clicked()
                    {
                        self.auto_play = !self.auto_play;
                        self.prev_instant = Instant::now();
                    }

                    if ui
                        .add(egui::Button::image(egui::Image::new(egui::include_image!(
                            "../icons/media-seek-forward.png"
                        ))))
                        .clicked()
                    {
                        if self.simulation.timeline_frames_count() > self.frame_index + 1 {
                            self.frame_index += 1;
                        } else {
                            self.frame_index = 0;
                        }
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

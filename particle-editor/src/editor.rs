use std::sync::Arc;

use wgpu::hal::Rect;
use winit::{
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::ActiveEventLoop,
    window::{Fullscreen, Window},
};

use crate::{
    backend::{Backend, BackendState, Packet},
    egui_utils::EguiContext,
    graphics::Graphics,
    simulation::Simulation,
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
    // autoplay related
    auto_play: bool,
    play_speed: f32,
    auto_play_counter: f32,
}

unsafe impl Send for Editor {}
unsafe impl Sync for Editor {}

impl Editor {
    pub async fn new(window: Arc<Window>) -> Editor {
        let gpu = WgpuContext::new(window).await;

        Editor {
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
            // Simulation Play related
            auto_play: false,
            play_speed: 1.,
            auto_play_counter: 0.,
        }
    }

    pub fn render(&mut self) {
        self.gpu.window.request_redraw();

        self.simulation.update(&mut self.backend);

        // ultra cutre but kinda works i guess
        if self.auto_play && self.simulation.timeline_frames_count() > 0 {
            if self.auto_play_counter > 60. {
                self.frame_index = (1 + self.frame_index) % self.simulation.timeline_frames_count();
                self.auto_play_counter = 0.;
            } else {
                self.auto_play_counter += self.play_speed;
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
            let out_state = editor.backend.backend_out_status().state;
            let in_state = editor.backend.backend_in_status().state;

            ui.collapsing(format!("Backend Output: {:?}", out_state), |ui| {
                ui.label(&editor.backend.backend_out_status().details);
            });
            ui.collapsing(format!("Backend Input: {:?}", in_state), |ui| {
                ui.label(&editor.backend.backend_in_status().details);
            });
            ui.add_space(5.);

            if ui.button("Connect by files").clicked() {
                editor.backend.open_backend_files();
            }

            ui.add_enabled_ui(false, |ui| {
                if ui.button("Connect by TCP").clicked() {
                    editor.backend.open_backend_files();
                }
            });

            if out_state == BackendState::Connected || in_state == BackendState::Connected {
                let button = ui.button("Disconnect");
                if button.clicked() {
                    editor.backend.close_connection();
                }
            }
        });

        self.ui_section(ui, "Simulation", |editor, ui| {
            if ui.button(if editor.auto_play { "Stop" } else { "Play" }).clicked() {
                editor.auto_play_counter = 0.;
                editor.auto_play = !editor.auto_play;
            }

            if ui.button("Clear Timeline").clicked() {
                editor.simulation.clear();
            }
            ui.horizontal(|ui| {
                let frames_count = editor.simulation.timeline_frames_count();
                let mut cursor = if editor.frame_index == 0 {
                    0
                } else {
                    frames_count.min(editor.frame_index + 1)
                };

                ui.label("Timeline");
                let st_range = frames_count.min(1);
                ui.add(
                    egui::Slider::new(&mut cursor, st_range..=frames_count)
                        .suffix(format!("/{}", frames_count)),
                );

                editor.frame_index = cursor.saturating_sub(1);
            });

            if ui.button(editor.play_speed.to_string() + "x").clicked() {
                editor.play_speed *= 2.;
                if editor.play_speed > 4. {
                    editor.play_speed = 0.25;
                }
            }
        });

        self.ui_section(ui, "Editor", |editor, ui| {
            ui.collapsing("Load Square", |ui| {
                ui.horizontal(|ui| {
                    ui.label("Size ");
                    ui.add(egui::Slider::new(&mut editor.box_size, 0..=20));
                });

                let can_send = editor.backend.backend_in_status().state == BackendState::Connected;
                ui.add_enabled_ui(can_send, |ui| {
                    if ui.button("Send To Backend").clicked() {
                        editor.backend.store(&Packet::square(editor.box_size));
                    }
                });
            });
        });

        self.ui_section(ui, "Render", |editor, ui| {
            ui.horizontal(|ui| {
                ui.label("Background Color");
                ui.color_edit_button_srgb(&mut editor.graphics.background_color);
            });

            ui.checkbox(&mut editor.graphics.uniform.rtx, "RTX");
        });
    }
}

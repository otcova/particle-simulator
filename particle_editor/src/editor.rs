use std::sync::Arc;

use egui::{
    CentralPanel, Color32, ComboBox, DragValue, FontId, Grid, Key, KeyboardShortcut, Margin,
    Modifiers, Pos2, RichText, ScrollArea, SidePanel, Slider, TextFormat, Vec2, WidgetText,
};
use particle_io::{Frame, FrameMetadata};
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
    box_size: u32,
    floating_windows: bool,
    close_window: bool,

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

        let mut editor = Editor {
            egui: EguiContext::new(&gpu, None, 1),
            graphics: Graphics::new(&gpu),
            simulation: Simulation::new(),
            backend: Backend::new(),
            gpu,

            ui_scale: 1.15,
            box_size: 5,
            floating_windows: false,
            close_window: false,

            num_formatter: NumFormatter {
                figures: 3,
                format: NumFormat::Scientific,
                rgb: [140, 140, 180],
            },

            sim_params: FrameMetadata::default(),

            // play related
            play_time: 0.,
            play_speed: 0.01,
            auto_play: false,
            loop_play: true,
        };

        editor.egui.context().style_mut(|style| {
            style.visuals.handle_shape = egui::style::HandleShape::Rect { aspect_ratio: 0.5 };
            style.spacing.scroll.bar_width = 6.;
            style.spacing.scroll.foreground_color = false;
        });

        editor.backend.open_tcp();
        egui_extras::install_image_loaders(editor.egui.context());
        editor
    }

    pub fn render(&mut self) {
        self.gpu.window.request_redraw();

        self.simulation.update(&mut self.backend);

        if self.auto_play {
            self.play_time += self.play_speed;
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
            .show(ctx, |ui| {
                let rect_points = ui.available_rect_before_wrap();

                let mut rect = egui::Rect::from_min_max(
                    rect_points.min * ui.pixels_per_point(),
                    rect_points.max * ui.pixels_per_point(),
                );

                rect.max = rect.max.min(egui::Pos2 {
                    x: (self.gpu.surface_size.width - 1) as f32,
                    y: (self.gpu.surface_size.height - 1) as f32,
                });

                let canvas_rect = self.graphics.canvas_size(Rect {
                    x: rect.min.x as u32,
                    y: rect.min.y as u32,
                    w: rect.size().x as u32,
                    h: rect.size().y as u32,
                });

                self.graphics.render(
                    &self.gpu,
                    encoder,
                    self.simulation.frame(self.play_time),
                    canvas_rect,
                );

                // let fill = ui.style().visuals.panel_fill;
                // ui.painter().rect_filled(_, 0, fill);
            });

        ctx.input_mut(|i| self.keyboard_shortcuts(i));
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

        self.ui_section(ui, "Editor", |editor, ui| {
            ui.collapsing("Load Square", |ui| {
                ui.horizontal(|ui| {
                    ui.label("Size ");
                    ui.add(Slider::new(&mut editor.box_size, 0..=20));
                });

                ui.add_enabled_ui(editor.backend.writer_connected(), |ui| {
                    if ui.button("Send To Backend").clicked() {
                        let mut frame = Frame::new();
                        *frame.metadata_mut() = editor.sim_params;
                        frame.push_square(editor.box_size);
                        editor.backend.write(&frame);
                    }
                });
            });
        });

        self.ui_section(ui, "Parameters", |editor, ui| {
            Grid::new("params-grid").num_columns(2).show(ui, |ui| {
                let mut params = editor.sim_params;

                ui.label("Step delta time");
                ui.add(Slider::new(&mut params.step_dt, 1e-10..=1e-1).logarithmic(true));
                ui.end_row();

                ui.label("Steps per frame");
                ui.add(Slider::new(&mut params.steps_per_frame, 1..=1000000).logarithmic(true));
                ui.end_row();

                ui.label("Frame delta time");
                ui.label(editor.num(0.000123, "s"));
                ui.end_row();

                ui.end_row();

                ui.label("Box size");
                ui.add(Slider::new(&mut editor.box_size, 0..=20));
                ui.end_row();

                ui.label("Data structure");
                ComboBox::from_id_salt("Data structure")
                    .selected_text("Compact Array")
                    .show_ui(ui, |ui| {
                        let mut v = 0;
                        ui.selectable_value(&mut v, 0, "Compact Array");
                        ui.selectable_value(&mut v, 1, "Matrix Buckets");
                    });
                ui.end_row();

                if editor.sim_params != params {
                    // Params changed => Send to backend
                }
                editor.sim_params = params;
            });
        });

        self.ui_section(ui, "Stats", |editor, ui| {
            Grid::new("stats-grid").num_columns(2).show(ui, |ui| {
                ui.label("Time");
                ui.label(editor.num(12.21311e-6, "s"));
                ui.end_row();

                ui.label("Step delta time");
                ui.label(editor.num(3.2113e-4, "s"));
                ui.end_row();

                ui.label("Num Particles");
                ui.label(editor.num(100., ""));
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
                if ram < 1000 {
                    ui.label(format!("{} B", ram));
                } else {
                    let mut f = editor.num_formatter;
                    f.figures = 3;
                    ui.label(f.fmt(ram as f32, "B"));
                }
                ui.end_row();
            });

            if ui.button("Clear Timeline").clicked() {
                editor.simulation.clear();
            }
        });

        self.ui_section(ui, "GUI", |editor, ui| {
            Grid::new("window-grid").num_columns(2).show(ui, |ui| {
                ui.label("GUI size ");
                ui.add(
                    DragValue::new(&mut editor.ui_scale)
                        .range(0.5..=3.0)
                        .prefix("x")
                        .speed(0.01),
                );
                ui.end_row();

                ui.label("Number Format ");
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
                            .suffix(format!("/{t_time_text}"))
                            .custom_formatter(|n, _| self.num_formatter.raw_string(n as f32, "s")),
                    );

                    ui.add_space(-resp.rect.width() - 8.);
                    ui.style_mut().spacing.slider_width =
                        (0 as f32).max(ui.available_width() - resp.rect.width());

                    ui.add(
                        egui::Slider::new(&mut cursor, (0.)..=t_time)
                            .suffix(format!("/{t_time_text}"))
                            .trailing_fill(self.simulation.sim_len() != 0.)
                            .custom_formatter(|n, _| self.num_formatter.raw_string(n as f32, "s")),
                    );

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

                    const MIN_SPEED: f32 = 1e-10;
                    const MAX_SPEED: f32 = 1e-1;

                    ui.style_mut().spacing.item_spacing =
                        Vec2::new(8.5 + ui.style().spacing.icon_width, 0.);
                    ui.add_space(
                        -(ui.style().spacing.slider_width + ui.style().spacing.icon_width + 8.),
                    );

                    ui.add(
                        egui::Slider::new(&mut self.play_speed, MIN_SPEED..=MAX_SPEED)
                            .custom_formatter(|n, _| self.num_formatter.raw_string(n as f32, "s"))
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NumFormat {
    Dashed,
    Scientific,
    Metric,
}

impl NumFormat {
    const fn name(self) -> &'static str {
        match self {
            NumFormat::Dashed => "Dashed",
            NumFormat::Scientific => "Scientific",
            NumFormat::Metric => "Metric",
        }
    }
}

#[derive(Clone, Copy)]
struct NumFormatter {
    format: NumFormat,
    figures: u32,
    rgb: [u8; 3],
}

impl NumFormatter {
    fn raw_string(&self, n: f32, unit: &'static str) -> String {
        let sign = if n < 0. { "-" } else { " " };
        match self.format {
            NumFormat::Dashed => {
                let decs = decimals_for_figures(n, self.figures);
                format!("{sign}{:.*} {unit}", decs, n)
            }
            NumFormat::Scientific => {
                let mut exp = n.log10();
                if exp == f32::NEG_INFINITY {
                    exp = 0.;
                }
                let exp = exp.floor() as i32;
                let mantissa = n / 10_f32.powi(exp);
                let text = format!("{sign}{:.*}·10", self.figures as usize - 1, mantissa);
                format!("{text}^{} {unit}", exp)
            }
            NumFormat::Metric => {
                const METRIC: &[(f32, &str)] = &[
                    (1e9, "G"),
                    (1e6, "M"),
                    (1e3, "k"),
                    (1.0, ""),
                    (1e-3, "m"),
                    (1e-6, "µ"),
                    (1e-9, "n"),
                    (1e-12, "p"),
                    (1e-15, "f"),
                ];

                let mut metric = *METRIC.last().unwrap();
                for &(divisor, suffix) in METRIC {
                    let threshold = divisor;
                    if n >= threshold {
                        metric = (divisor, suffix);
                        break;
                    }
                }

                let scaled = n / metric.0;
                let decs = decimals_for_figures(scaled, self.figures);
                format!("{sign}{:.*} {}{unit}", decs, scaled, metric.1)
            }
        }
    }

    fn fmt(&self, n: f32, unit: &'static str) -> WidgetText {
        let sign = if n < 0. { "-" } else { " " };
        let n = n.abs();

        let figs = self.figures as usize;
        let color = Color32::from_rgb(self.rgb[0], self.rgb[1], self.rgb[2]);
        let font_id = FontId::monospace(12.);
        let exp_font = FontId::monospace(10.);

        match self.format {
            NumFormat::Dashed => {
                let decs = decimals_for_figures(n, self.figures);
                RichText::new(format!("{sign}{:.*} {unit}", decs, n))
                    .color(color)
                    .font(font_id)
                    .into()
            }
            NumFormat::Scientific => {
                let mut exp = n.log10();
                if exp == f32::NEG_INFINITY {
                    exp = 0.;
                }
                let exp = exp.floor() as i32;
                let mantissa = n / 10_f32.powi(exp);

                let mut text = egui::text::LayoutJob::default();
                text.append(
                    &format!("{sign}{:.*}·10", figs - 1, mantissa),
                    0.,
                    TextFormat {
                        color,
                        font_id: font_id.clone(),
                        ..Default::default()
                    },
                );
                text.append(
                    &format!("{} ", exp),
                    0.,
                    TextFormat {
                        color,
                        valign: egui::Align::TOP,
                        font_id: exp_font,
                        ..Default::default()
                    },
                );
                text.append(
                    unit,
                    0.,
                    TextFormat {
                        color,
                        font_id,
                        ..Default::default()
                    },
                );
                text.into()
            }
            NumFormat::Metric => {
                const METRIC: &[(f32, &str)] = &[
                    (1e9, "G"),
                    (1e6, "M"),
                    (1e3, "k"),
                    (1.0, ""),
                    (1e-3, "m"),
                    (1e-6, "µ"),
                    (1e-9, "n"),
                    (1e-12, "p"),
                    (1e-15, "f"),
                ];

                let mut metric = *METRIC.last().unwrap();
                for &(divisor, suffix) in METRIC {
                    let threshold = divisor;
                    if n >= threshold {
                        metric = (divisor, suffix);
                        break;
                    }
                }

                let scaled = n / metric.0;
                let decs = decimals_for_figures(scaled, self.figures);
                RichText::new(format!("{sign}{:.*} {}{unit}", decs, scaled, metric.1))
                    .color(color)
                    .font(font_id)
                    .into()
            }
        }
    }
}

fn decimals_for_figures(n: f32, sign_figures: u32) -> usize {
    let exp10 = n.abs().log10();
    if exp10 == f32::NEG_INFINITY {
        return 0;
    }
    let digits = exp10.floor() as isize + 1;
    (sign_figures as isize - digits).max(0) as usize
}

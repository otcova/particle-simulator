use std::{env, sync::Arc};

use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{WindowAttributes, WindowId},
};

use crate::editor::Editor;

mod backend;
mod editor;
mod egui_utils;
mod graphics;
mod simulation;
mod wgpu_utils;

#[derive(Default)]
struct App {
    editor: Option<Editor>,
    visible: bool,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let fullscreen = env::var("WINIT_FULLSCREEN").ok().is_some_and(|s| s == "y");

        let attributes = WindowAttributes::default()
            .with_visible(false)
            .with_title("Particle Editor");
        let window = Arc::new(event_loop.create_window(attributes).unwrap());
        window.request_redraw();

        let editor = pollster::block_on(Editor::new(window));
        if fullscreen {
            editor.toggle_fullscreen();
        }
        self.editor = Some(editor);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let editor = self.editor.as_mut().unwrap();
        editor.handle_input(event_loop, &event);
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                editor.render();
            }
            WindowEvent::Resized(size) => {
                editor.resize(size);
            }
            _ => (),
        }

        if !self.visible {
            editor.window().set_visible(true);
            self.visible = true;
        }
    }
}

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new().expect("Unable to start event loop");

    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::default();
    event_loop.run_app(&mut app).unwrap();
}

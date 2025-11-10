use std::sync::Arc;

use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    platform::x11::EventLoopBuilderExtX11,
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
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attributes = WindowAttributes::default();
        // attributes.inner_size = Some(LogicalSize::new(1024, 768).into());
        let window = Arc::new(event_loop.create_window(attributes).unwrap());

        let editor = pollster::block_on(Editor::new(window));
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
    }
}

fn main() {
    env_logger::init();

    let event_loop = if cfg!(target_os = "linux") {
        // While this issue is unsolved: https://github.com/rust-windowing/winit/issues/4267
        // we deafult to x11.
        // EventLoop::builder().with_x11().build().unwrap()
        EventLoop::builder().with_x11().build()
    } else {
        EventLoop::new()
    }
    .expect("Unable to start event loop");

    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::default();
    event_loop.run_app(&mut app).unwrap();
}

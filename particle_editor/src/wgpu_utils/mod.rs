use std::{borrow::Cow, sync::Arc};

use wgpu::rwh::{HasWindowHandle, RawWindowHandle};
use winit::{dpi::PhysicalSize, window::Window};

pub struct WgpuContext {
    pub window: Arc<Window>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface_size: winit::dpi::PhysicalSize<u32>,
    surface: wgpu::Surface<'static>,
    pub surface_format: wgpu::TextureFormat,

    pub surface_texture: Option<wgpu::SurfaceTexture>,
    pub surface_view: Option<wgpu::TextureView>,
}

impl WgpuContext {
    pub async fn new(window: Arc<Window>) -> WgpuContext {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .unwrap();
        log::info!(
            "Created a {} window with {:?} {:?}",
            match window.window_handle().map(|h| h.as_raw()) {
                Err(error) => Cow::Owned(error.to_string()),
                Ok(RawWindowHandle::Wayland(_)) => "Wayland".into(),
                Ok(RawWindowHandle::Xlib(_)) => "X11 (Xlib)".into(),
                Ok(RawWindowHandle::Xcb(_)) => "X11 (XCB)".into(),
                Ok(RawWindowHandle::Win32(_)) => "Win32".into(),
                Ok(raw) => format!("{:?}", raw).into(),
            },
            adapter.get_info().backend,
            adapter.get_info().device_type,
        );

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();

        let surface_size = window.inner_size();

        let cap = surface.get_capabilities(&adapter);
        //let surface_format = cap.formats[0].add_srgb_suffix();
        let surface_format = cap
            .formats
            .first()
            .expect("Unable to create a surface")
            .remove_srgb_suffix();
        // add_srgb_suffix();

        Self::configure_surface(&device, &surface, surface_format, surface_size);

        WgpuContext {
            window,
            device,
            queue,
            surface_size,
            surface,
            surface_format,

            surface_texture: None,
            surface_view: None,
        }
    }

    fn configure_surface(
        device: &wgpu::Device,
        surface: &wgpu::Surface,
        format: wgpu::TextureFormat,
        size: PhysicalSize<u32>,
    ) {
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            view_formats: vec![format.add_srgb_suffix()],
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            width: size.width,
            height: size.height,
            desired_maximum_frame_latency: 1,
            present_mode: wgpu::PresentMode::AutoVsync,
        };
        surface.configure(device, &surface_config);
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.surface_size = new_size;
        Self::configure_surface(
            &self.device,
            &self.surface,
            self.surface_format,
            self.surface_size,
        );
    }

    pub fn start_frame(&mut self) {
        let surface_texture = match self.surface.get_current_texture() {
            Ok(texture) => texture,
            Err(_) => {
                Self::configure_surface(
                    &self.device,
                    &self.surface,
                    self.surface_format,
                    self.surface_size,
                );
                self.surface
                    .get_current_texture()
                    .expect("Failed to acquire next swapchain texture")
            }
        };

        self.surface_view = Some(surface_texture.texture.create_view(
            &wgpu::TextureViewDescriptor {
                format: Some(self.surface_format),
                ..Default::default()
            },
        ));

        self.surface_texture = Some(surface_texture);
    }

    pub fn end_frame(&mut self) {
        self.window.pre_present_notify();
        self.surface_texture
            .take()
            .expect("Must start frame before end_frame")
            .present();
    }
}

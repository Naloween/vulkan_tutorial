use vulkano::{
    instance::{Instance, InstanceCreateInfo},
    Version, VulkanLibrary,
};
use vulkano_win::VkSurfaceBuild;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

pub mod graphic_engine;

pub struct App {
    event_loop: EventLoop<()>,
    graphic_engine: graphic_engine::Graphicengine,
}

impl App {
    pub fn new() -> App {
        // Vulkan instance
        let instance = {
            let library = VulkanLibrary::new().unwrap();
            let extensions = vulkano_win::required_extensions(&library);

            Instance::new(
                library,
                InstanceCreateInfo {
                    enabled_extensions: extensions,
                    enumerate_portability: true, // required for MoltenVK on macOS
                    max_api_version: Some(Version::V1_1),
                    ..Default::default()
                },
            )
            .unwrap()
        };
        // Vulkan surface on a window
        let event_loop = EventLoop::new();
        let surface = WindowBuilder::new()
            .build_vk_surface(&event_loop, instance.clone())
            .unwrap();

        let graphic_engine = graphic_engine::Graphicengine::new(instance, surface);

        return App {
            event_loop,
            graphic_engine,
        };
    }

    pub fn run(mut self) {
        let mut recreate_swapchain = false;
        self.event_loop.run(move |event, _, control_flow| {
            match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => {
                    *control_flow = ControlFlow::Exit;
                }
                Event::WindowEvent {
                    event: WindowEvent::Resized(_),
                    ..
                } => {
                    recreate_swapchain = true;
                }
                Event::RedrawEventsCleared => {
                    self.graphic_engine.render(&mut recreate_swapchain);
                }
                _ => {}
            }

            if recreate_swapchain {
                self.graphic_engine
                    .recreate_swapchain(&mut recreate_swapchain);
            }
        });
    }
}

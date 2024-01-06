use std::sync::Arc;

use vulkano::{
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        RenderPassBeginInfo, SubpassContents,
    },
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo,
    },
    format::ClearValue,
    image::{view::ImageView, ImageAccess, SwapchainImage},
    instance::Instance,
    pipeline::graphics::viewport::Viewport,
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass},
    swapchain::{
        self, AcquireError, Surface, Swapchain, SwapchainCreateInfo, SwapchainCreationError,
        SwapchainPresentInfo,
    },
    sync::{self, FlushError, GpuFuture},
};

use winit::window::Window;

pub struct Graphicengine {
    surface: Arc<Surface>,
    device: Arc<Device>,
    swapchain: Arc<Swapchain>,
    queue: Arc<Queue>,
    framebuffers: Vec<Arc<Framebuffer>>,
    command_buffer_allocator: StandardCommandBufferAllocator,
    render_pass: Arc<RenderPass>,
    viewport: Viewport,

    previous_frame_end: Option<Box<dyn GpuFuture>>,
}

impl Graphicengine {
    pub fn new(instance: Arc<Instance>, surface: Arc<Surface>) -> Graphicengine {
        // Finding device
        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };
        let (physical_device, queue_family_index) =
            Graphicengine::get_best_compatible_physical_device(
                instance,
                &device_extensions,
                &surface,
            );
        let (device, mut queues) =
            Graphicengine::get_device(physical_device, queue_family_index, device_extensions);

        let queue = queues.next().unwrap();

        // Getting swapchain
        let (swapchain, images) = Graphicengine::get_swapchain(&device, &surface);

        // Allocator
        let command_buffer_allocator =
            StandardCommandBufferAllocator::new(device.clone(), Default::default());
        // Render pass
        let render_pass = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: swapchain.image_format(),
                    samples: 1,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {}
            }
        )
        .unwrap();

        // Viewport
        let mut viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: [0.0, 0.0],
            depth_range: 0.0..1.0,
        };

        // Framebuffer
        let framebuffers =
            Graphicengine::window_size_dependent_setup(&images, render_pass.clone(), &mut viewport);

        // for renders
        let previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<dyn GpuFuture>);

        Graphicengine {
            device,
            surface,
            swapchain,
            queue,
            framebuffers,
            command_buffer_allocator,
            render_pass,
            viewport,
            previous_frame_end,
        }
    }

    pub fn render(&mut self, recreate_swapchain: &mut bool) {
        self.previous_frame_end
            .as_mut()
            .take()
            .unwrap()
            .cleanup_finished();

        let (image_index, suboptimal, acquire_future) =
            match swapchain::acquire_next_image(self.swapchain.clone(), None) {
                Ok(r) => r,
                Err(AcquireError::OutOfDate) => {
                    *recreate_swapchain = true;
                    return;
                }
                Err(e) => panic!("Failed to acquire next image: {:?}", e),
            };

        if suboptimal {
            *recreate_swapchain = true;
        }

        let clear_values: Vec<Option<ClearValue>> =
            vec![Some(ClearValue::Float([0.0, 0.68, 1.0, 1.0]))];

        let mut cmd_buffer_builder = AutoCommandBufferBuilder::primary(
            &self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        cmd_buffer_builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values,
                    ..RenderPassBeginInfo::framebuffer(
                        self.framebuffers[image_index as usize].clone(),
                    )
                },
                SubpassContents::Inline,
            )
            .unwrap()
            .end_render_pass()
            .unwrap();

        let command_buffer = cmd_buffer_builder.build().unwrap();

        let future = self
            .previous_frame_end
            .take()
            .unwrap()
            .join(acquire_future)
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(
                self.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(self.swapchain.clone(), image_index),
            )
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                self.previous_frame_end = Some(Box::new(future) as Box<_>);
            }
            Err(FlushError::OutOfDate) => {
                *recreate_swapchain = true;
                self.previous_frame_end = Some(Box::new(sync::now(self.device.clone())) as Box<_>);
            }
            Err(e) => {
                println!("Failed to flush future: {:?}", e);
                self.previous_frame_end = Some(Box::new(sync::now(self.device.clone())) as Box<_>);
            }
        }
    }

    pub fn recreate_swapchain(&mut self, recreate_swapchain: &mut bool) {
        let window = self
            .surface
            .object()
            .unwrap()
            .downcast_ref::<Window>()
            .unwrap();
        let image_extent: [u32; 2] = window.inner_size().into();

        let (new_swapchain, new_images) = match self.swapchain.recreate(SwapchainCreateInfo {
            image_extent,
            ..self.swapchain.create_info()
        }) {
            Ok(r) => r,
            Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
            Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
        };

        self.swapchain = new_swapchain;
        self.framebuffers = Graphicengine::window_size_dependent_setup(
            &new_images,
            self.render_pass.clone(),
            &mut self.viewport,
        );
        *recreate_swapchain = false;
    }

    fn get_best_compatible_physical_device(
        instance: Arc<Instance>,
        device_extensions: &DeviceExtensions,
        surface: &Arc<Surface>,
    ) -> (Arc<PhysicalDevice>, u32) {
        instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|physical_device| {
                physical_device
                    .supported_extensions()
                    .contains(&device_extensions)
            })
            .filter_map(|physical_device| {
                physical_device
                    .queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, queue_family)| {
                        // pick first queue_familiy_index that handles graphics and can draw on the surface created by winit
                        queue_family.queue_flags.graphics
                            && physical_device
                                .surface_support(i as u32, surface)
                                .unwrap_or(false)
                    })
                    .map(|i| (physical_device, i as u32))
            })
            .min_by_key(|(p, _)| {
                // lower score for preferred device types
                match p.properties().device_type {
                    PhysicalDeviceType::DiscreteGpu => 0,
                    PhysicalDeviceType::IntegratedGpu => 1,
                    PhysicalDeviceType::VirtualGpu => 2,
                    PhysicalDeviceType::Cpu => 3,
                    PhysicalDeviceType::Other => 4,
                    _ => 5,
                }
            })
            .expect("No suitable physical device found")
    }

    fn get_device(
        physical_device: Arc<PhysicalDevice>,
        queue_family_index: u32,
        device_extensions: DeviceExtensions,
    ) -> (Arc<Device>, impl ExactSizeIterator<Item = Arc<Queue>>) {
        Device::new(
            physical_device,
            DeviceCreateInfo {
                enabled_extensions: device_extensions,
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .unwrap()
    }

    fn get_swapchain(
        device: &Arc<Device>,
        surface: &Arc<Surface>,
    ) -> (Arc<Swapchain>, Vec<Arc<SwapchainImage>>) {
        let caps = device
            .physical_device()
            .surface_capabilities(surface, Default::default())
            .unwrap();

        let usage = caps.supported_usage_flags;
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();

        let image_format = Some(
            device
                .physical_device()
                .surface_formats(surface, Default::default())
                .unwrap()[1]
                .0,
        );

        let window = surface.object().unwrap().downcast_ref::<Window>().unwrap();
        let image_extent: [u32; 2] = window.inner_size().into();

        Swapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count: caps.min_image_count,
                image_format,
                image_extent,
                image_usage: usage,
                composite_alpha: alpha,
                ..Default::default()
            },
        )
        .unwrap()
    }

    fn window_size_dependent_setup(
        images: &[Arc<SwapchainImage>],
        render_pass: Arc<RenderPass>,
        viewport: &mut Viewport,
    ) -> Vec<Arc<Framebuffer>> {
        let dimensions = images[0].dimensions().width_height();
        viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];

        images
            .iter()
            .map(|image| {
                let view = ImageView::new_default(image.clone()).unwrap();
                Framebuffer::new(
                    render_pass.clone(),
                    FramebufferCreateInfo {
                        attachments: vec![view],
                        ..Default::default()
                    },
                )
                .unwrap()
            })
            .collect::<Vec<_>>()
    }
}

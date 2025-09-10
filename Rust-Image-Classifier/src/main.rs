use burn::optim::{SgdConfig, momentum::MomentumConfig};
use custom_image_dataset::training::TrainingConfig;

// Import only when backend features are enabled
#[cfg(feature = "wgpu")]
use {burn::backend::Autodiff, custom_image_dataset::training::train};

/// Creates a training configuration with SGD optimizer and momentum.
fn create_config() -> TrainingConfig {
    TrainingConfig::new(SgdConfig::new().with_momentum(Some(MomentumConfig {
        momentum: 0.9,
        dampening: 0.,
        nesterov: false,
    })))
}

fn main() {
    #[allow(unused_variables)]
    let config = create_config();

    #[cfg(feature = "wgpu")]
    {
        use burn::backend::wgpu::{Wgpu, WgpuDevice};
        train::<Autodiff<Wgpu>>(config, WgpuDevice::default());
    }
}

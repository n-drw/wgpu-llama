pub(crate) mod cache;
pub mod llama;
pub mod pretrained;
pub mod sampling;
pub mod tokenizer;
pub mod app;
mod transformer;

#[cfg(test)]
mod tests {
    #[cfg(feature = "cuda")]
    use burn::{backend::cuda::Cuda, tensor::f16};
    #[cfg(feature = "cuda")]
    pub type TestBackend = Cuda<f16, i32>;

    #[cfg(feature = "wgpu")]
    use burn::{backend::wgpu::Wgpu, tensor::f16};
    #[cfg(feature = "wgpu")] 
    pub type TestBackend = Wgpu<f32, i32>;

    // NOTE: no tests on tch cpu (f32)
    #[cfg(feature = "tch-gpu")]
    use burn::{backend::LibTorch, tensor::f16};
    #[cfg(feature = "tch-gpu")]
    pub type TestBackend = LibTorch<f16>;

    #[cfg(any(feature = "cuda", feature = "tch-gpu"))]
    pub type TestTensor<const D: usize> = burn::tensor::Tensor<TestBackend, D>;
}

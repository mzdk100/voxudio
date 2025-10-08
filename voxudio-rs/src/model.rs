mod asr;
mod see;
mod tcc;
mod vad;

use {
    crate::OperationError,
    ort::session::{Session, builder::SessionBuilder},
};
pub use {asr::*, see::*, tcc::*, vad::*};

fn get_session_builder() -> Result<SessionBuilder, OperationError> {
    let builder = Session::builder()?;
    #[cfg(target_os = "android")]
    let builder = builder.with_execution_providers(&[
        ort::execution_providers::WebGPUExecutionProvider::default().build(),
    ])?;

    Ok(builder)
}

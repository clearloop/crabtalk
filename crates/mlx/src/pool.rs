//! `MlxPool` — Rust-side safe wrapper around the Swift pool FFI.
//!
//! The actual multi-model cache, idle eviction, and model loading live
//! in Swift (see `mlx/Sources/CrabllmMlx/Pool.swift`). This module is
//! a thin `NonNull` handle with `Send + Sync` and `Drop`.

use crate::ffi;
use crate::session::{
    OwnedRequest, ResultGuard, copy_c_string_opt, take_owned_c_string, translate_status,
};
use crabllm_core::Error;
use std::{
    ffi::{CString, c_char},
    os::raw::{c_int, c_void},
    panic, ptr,
};

/// Handle to a Swift-side multi-model pool.
pub struct MlxPool {
    inner: ptr::NonNull<ffi::CrabllmMlxPool>,
}

unsafe impl Send for MlxPool {}
unsafe impl Sync for MlxPool {}

impl MlxPool {
    /// Create a pool with idle eviction. `idle_timeout_secs == 0` uses
    /// the Swift default (30 min).
    pub fn new(idle_timeout_secs: u64) -> Result<Self, Error> {
        let mut pool_ptr: *mut ffi::CrabllmMlxPool = ptr::null_mut();
        let mut err_ptr: *mut c_char = ptr::null_mut();
        let status =
            unsafe { ffi::crabllm_mlx_pool_new(idle_timeout_secs, &mut pool_ptr, &mut err_ptr) };
        if status == ffi::CRABLLM_MLX_OK {
            let inner = ptr::NonNull::new(pool_ptr).ok_or_else(|| {
                Error::Internal("mlx: pool_new OK but pointer is NULL".to_string())
            })?;
            Ok(MlxPool { inner })
        } else {
            let msg = unsafe { take_owned_c_string(err_ptr) };
            Err(translate_status(status, msg))
        }
    }

    /// Non-streaming generation through the pool.
    pub fn generate(
        &self,
        model_dir: &str,
        req: &crate::session::GenerateRequest<'_>,
    ) -> Result<crate::session::GenerateOutput, Error> {
        let model_c = CString::new(model_dir)
            .map_err(|_| Error::Internal("mlx: model_dir contains NUL byte".to_string()))?;
        let owned = OwnedRequest::new(req)?;
        let mut guard = ResultGuard::zeroed();
        let status = unsafe {
            ffi::crabllm_mlx_pool_generate(
                self.inner.as_ptr(),
                model_c.as_ptr(),
                owned.as_raw(),
                guard.as_mut_ptr(),
            )
        };
        if status == ffi::CRABLLM_MLX_OK {
            let text = unsafe { copy_c_string_opt(guard.inner.text)? }.ok_or_else(|| {
                Error::Internal("mlx: pool generate OK but result.text is NULL".to_string())
            })?;
            let tool_calls_json = unsafe { copy_c_string_opt(guard.inner.tool_calls_json)? };
            Ok(crate::session::GenerateOutput {
                text,
                tool_calls_json,
                prompt_tokens: guard.inner.prompt_tokens,
                completion_tokens: guard.inner.completion_tokens,
            })
        } else {
            let msg = unsafe { copy_c_string_opt(guard.inner.error) }
                .ok()
                .flatten()
                .unwrap_or_else(|| "(no error message from Swift)".to_string());
            Err(translate_status(status, msg))
        }
    }

    /// Streaming generation through the pool.
    pub fn generate_stream<F>(
        &self,
        model_dir: &str,
        req: &crate::session::GenerateRequest<'_>,
        mut on_token: F,
    ) -> Result<crate::session::StreamOutput, Error>
    where
        F: FnMut(&str) -> bool,
    {
        let model_c = CString::new(model_dir)
            .map_err(|_| Error::Internal("mlx: model_dir contains NUL byte".to_string()))?;
        let owned = OwnedRequest::new(req)?;

        struct TrampolineState<'cb, F: FnMut(&str) -> bool> {
            cb: &'cb mut F,
            panicked: bool,
        }

        extern "C" fn trampoline<F: FnMut(&str) -> bool>(
            token: *const c_char,
            user_data: *mut c_void,
        ) -> c_int {
            if user_data.is_null() {
                return 1;
            }
            let state = unsafe { &mut *(user_data as *mut TrampolineState<'_, F>) };
            if state.panicked {
                return 1;
            }
            if token.is_null() {
                return 0;
            }
            let slice = unsafe { std::ffi::CStr::from_ptr(token) };
            let s = match slice.to_str() {
                Ok(s) => s,
                Err(_) => return 1,
            };
            let cb = &mut state.cb;
            let result = panic::catch_unwind(panic::AssertUnwindSafe(|| (*cb)(s)));
            match result {
                Ok(true) => 1,
                Ok(false) => 0,
                Err(_) => {
                    state.panicked = true;
                    1
                }
            }
        }

        let mut state = TrampolineState {
            cb: &mut on_token,
            panicked: false,
        };
        let mut guard = ResultGuard::zeroed();
        let status = unsafe {
            ffi::crabllm_mlx_pool_generate_stream(
                self.inner.as_ptr(),
                model_c.as_ptr(),
                owned.as_raw(),
                trampoline::<F>,
                &mut state as *mut TrampolineState<'_, F> as *mut c_void,
                guard.as_mut_ptr(),
            )
        };

        if state.panicked {
            return Err(Error::Internal(
                "mlx: token callback panicked during streaming".to_string(),
            ));
        }

        if status == ffi::CRABLLM_MLX_OK {
            let tool_calls_json = unsafe { copy_c_string_opt(guard.inner.tool_calls_json)? };
            Ok(crate::session::StreamOutput {
                tool_calls_json,
                prompt_tokens: guard.inner.prompt_tokens,
                completion_tokens: guard.inner.completion_tokens,
            })
        } else {
            let msg = unsafe { copy_c_string_opt(guard.inner.error) }
                .ok()
                .flatten()
                .unwrap_or_else(|| "(no error message from Swift)".to_string());
            Err(translate_status(status, msg))
        }
    }

    /// Evict a single model.
    pub fn evict(&self, model_dir: &str) {
        if let Ok(c) = CString::new(model_dir) {
            unsafe { ffi::crabllm_mlx_pool_evict(self.inner.as_ptr(), c.as_ptr()) };
        }
    }

    /// Evict all models and stop the idle monitor.
    pub fn stop_all(&self) {
        unsafe { ffi::crabllm_mlx_pool_stop_all(self.inner.as_ptr()) };
    }
}

impl Drop for MlxPool {
    fn drop(&mut self) {
        unsafe { ffi::crabllm_mlx_pool_free(self.inner.as_ptr()) };
    }
}

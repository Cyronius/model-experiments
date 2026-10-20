//! C interface. See include/titler.h for the contract.

use std::cell::RefCell;
use std::ffi::{c_char, c_int, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::slice;

pub struct TitlerHandle(titler::Titler);

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn set_error(message: impl Into<Vec<u8>>) {
    let c = CString::new(message).unwrap_or_else(|_| CString::new("error").unwrap());
    LAST_ERROR.with(|e| *e.borrow_mut() = c);
}

fn guard<T>(fail: T, f: impl FnOnce() -> Result<T, String>) -> T {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            set_error(e);
            fail
        }
        Err(_) => {
            set_error("internal error (panic)");
            fail
        }
    }
}

fn boxed(t: titler::Result<titler::Titler>) -> Result<*mut TitlerHandle, String> {
    t.map(|t| Box::into_raw(Box::new(TitlerHandle(t)))).map_err(|e| e.to_string())
}

/// # Safety
/// `path` must be a valid NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn titler_new_from_file(path: *const c_char) -> *mut TitlerHandle {
    guard(std::ptr::null_mut(), || {
        if path.is_null() {
            return Err("path is NULL".into());
        }
        let path = CStr::from_ptr(path).to_string_lossy().into_owned();
        boxed(titler::Titler::from_path(path))
    })
}

/// # Safety
/// `data` must point to `len` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn titler_new_from_memory(data: *const u8, len: usize) -> *mut TitlerHandle {
    guard(std::ptr::null_mut(), || {
        if data.is_null() {
            return Err("data is NULL".into());
        }
        boxed(titler::Titler::from_gguf_bytes(slice::from_raw_parts(data, len)))
    })
}

/// # Safety
/// `t` must come from a `titler_new_*` call and not be used afterwards. NULL is allowed.
#[no_mangle]
pub unsafe extern "C" fn titler_free(t: *mut TitlerHandle) {
    if !t.is_null() {
        drop(Box::from_raw(t));
    }
}

unsafe fn title_into(t: *const TitlerHandle, prompt: &[u8], beams: c_int, out: *mut c_char, cap: usize) -> i64 {
    guard(-1, || {
        let handle = t.as_ref().ok_or("titler handle is NULL")?;
        let beams = if beams <= 0 { titler::DEFAULT_BEAMS } else { beams as usize };
        let title = handle.0.title_with_beams(&String::from_utf8_lossy(prompt), beams).map_err(|e| e.to_string())?;
        if !out.is_null() && cap > 0 {
            let mut n = title.len().min(cap - 1);
            while !title.is_char_boundary(n) {
                n -= 1;
            }
            std::ptr::copy_nonoverlapping(title.as_ptr(), out as *mut u8, n);
            *out.add(n) = 0;
        }
        Ok(title.len() as i64)
    })
}

/// # Safety
/// `t` is a live handle, `prompt` a NUL-terminated string, `out` has room for `cap` bytes.
#[no_mangle]
pub unsafe extern "C" fn titler_title(t: *const TitlerHandle, prompt: *const c_char, beams: c_int, out: *mut c_char, cap: usize) -> i64 {
    if prompt.is_null() {
        set_error("prompt is NULL");
        return -1;
    }
    title_into(t, CStr::from_ptr(prompt).to_bytes(), beams, out, cap)
}

/// # Safety
/// `t` is a live handle, `prompt` has `len` readable bytes, `out` has room for `cap` bytes.
#[no_mangle]
pub unsafe extern "C" fn titler_title_n(t: *const TitlerHandle, prompt: *const u8, len: usize, beams: c_int, out: *mut c_char, cap: usize) -> i64 {
    if prompt.is_null() && len > 0 {
        set_error("prompt is NULL");
        return -1;
    }
    let bytes = if len == 0 { &[][..] } else { slice::from_raw_parts(prompt, len) };
    title_into(t, bytes, beams, out, cap)
}

#[no_mangle]
pub extern "C" fn titler_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}

#[no_mangle]
pub extern "C" fn titler_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

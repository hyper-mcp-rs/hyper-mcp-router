//! Unit-test-only helpers shared across the crate's test modules. Compiled
//! exclusively under `cfg(test)` (see `lib.rs`); never part of the shipped
//! binary.

use std::sync::{Arc, Mutex};

/// A cloneable in-memory sink for `tracing_subscriber::fmt`, so tests can
/// assert on exactly what a log event emitted.
#[derive(Clone, Default)]
pub struct CaptureWriter(Arc<Mutex<Vec<u8>>>);

impl std::io::Write for CaptureWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for CaptureWriter {
    type Writer = CaptureWriter;
    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

/// Run `f` under a thread-local subscriber capped at `max_level` and return
/// everything it logged.
pub fn captured_log(max_level: tracing::Level, f: impl FnOnce()) -> String {
    let writer = CaptureWriter::default();
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(max_level)
        .with_writer(writer.clone())
        .with_ansi(false)
        .finish();
    tracing::subscriber::with_default(subscriber, f);
    let bytes = writer.0.lock().unwrap().clone();
    String::from_utf8(bytes).unwrap()
}

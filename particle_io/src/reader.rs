use bytemuck::{bytes_of_mut, cast_slice_mut};

use crate::{FrameHeader, particle::Frame};
use std::{
    fs::OpenOptions,
    io::{ErrorKind, Read},
    path::Path,
    sync::mpsc::{self, Receiver, TryRecvError},
    time::Duration,
};

pub struct Reader {
    receiver: Receiver<Option<Frame>>,
}

impl Reader {
    pub fn new<R: Read + Send + 'static>(mut stream: R) -> Reader {
        let (sender, receiver) = mpsc::sync_channel(16);

        std::thread::spawn(move || {
            let mut header = FrameHeader::default();
            loop {
                let abort = || sender.send(None).is_err();

                // Read header
                header.set_invalid_signature();
                let raw_header = bytes_of_mut(&mut header);
                if read_blocking(&mut stream, raw_header, abort).is_err() {
                    break;
                }

                if !header.is_valid() {
                    eprintln!("Read frame with invalid signature");
                    continue;
                }

                // Read Body
                let mut frame = Frame::from_header(header);
                let raw_particles = cast_slice_mut(frame.particles_mut());
                if read_blocking(&mut stream, raw_particles, abort).is_err() {
                    break;
                }

                // Send to main thread
                if sender.send(Some(frame)).is_err() {
                    break;
                }
            }
        });

        Reader { receiver }
    }

    pub fn open_file<P: AsRef<Path>>(path: P) -> Result<Reader, String> {
        match OpenOptions::new().read(true).open(path) {
            Ok(file) => Ok(Self::new(file)),
            Err(error) => Err(format!("{}", error)),
        }
    }

    // Returns error if disconnected
    // Returns Ok(None) if connected but waiting for more data
    #[allow(clippy::result_unit_err)]
    pub fn read(&self) -> Result<Option<Frame>, ()> {
        loop {
            match self.receiver.try_recv() {
                Ok(None) => continue,
                Ok(Some(frame)) => return Ok(Some(frame)),
                Err(TryRecvError::Empty) => return Ok(None),
                Err(TryRecvError::Disconnected) => return Err(()),
            }
        }
    }
}

fn read_blocking<R: Read, F: FnMut() -> bool>(
    mut file: R,
    mut buf: &mut [u8],
    mut abort: F,
) -> std::io::Result<()> {
    let mut retry = move || {
        let abort_error = std::io::Error::new(
            ErrorKind::ConnectionAborted,
            "Read canceled due to aborted connection",
        );

        if abort() {
            return Err(abort_error);
        }

        std::thread::sleep(Duration::from_millis(1));

        if abort() {
            return Err(abort_error);
        }
        Ok(())
    };

    while !buf.is_empty() {
        match file.read(buf) {
            Ok(0) => retry()?,
            Ok(n) => buf = &mut buf[n..],
            Err(ref e) if e.kind() == ErrorKind::UnexpectedEof => retry()?,
            Err(ref e) if e.kind() == ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_reader() {
        let mut frame1 = Frame::new();
        let mut frame2 = Frame::new();
        let mut frame3 = Frame::new();

        frame1.push_square(5);
        frame2.push_square(21);
        frame3.push_square(2);

        let mut raw_data = Vec::new();
        raw_data.extend_from_slice(frame1.bytes());
        raw_data.extend_from_slice(frame2.bytes());
        raw_data.extend_from_slice(frame3.bytes());

        let stream = Cursor::new(raw_data);
        let reader = Reader::new(stream);

        // Thread should not have started yet
        assert!(reader.read() == Ok(None));

        // Give time for the thread to start up
        std::thread::sleep(Duration::from_millis(100));

        assert!(reader.read() == Ok(Some(frame1)));
        assert!(reader.read() == Ok(Some(frame2)));
        assert!(reader.read() == Ok(Some(frame3)));
        assert!(reader.read() == Ok(None));
    }
}

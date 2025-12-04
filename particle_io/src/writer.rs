use crate::Frame;
use std::{fs, io, path::Path};

pub struct Writer {
    stream: Box<dyn io::Write>,
}

impl Writer {
    pub fn new<W: io::Write + 'static>(stream: W) -> Writer {
        Writer {
            stream: Box::new(stream),
        }
    }

    // Buffer will only be used for start/end_write.
    pub fn open_file<P: AsRef<Path>>(path: P) -> io::Result<Writer> {
        let file = fs::OpenOptions::new().append(true).open(path)?;
        Ok(Writer::new(file))
    }

    // This does an extra copy wich could be avoided using write_fn
    // and writing directly to the buffer.
    pub fn write(&mut self, frame: &Frame) -> Result<(), String> {
        self.stream
            .write_all(frame.bytes())
            .map_err(|err| format!("{}", err))
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use std::io::{Read, pipe};
//
//     #[test]
//     fn test_writer() {
//         let mut frame1 = Frame::new();
//         let mut frame2 = Frame::new();
//         let mut frame3 = Frame::new();
//
//         let pos = (0.5, 0.5);
//         frame1.push_square(pos, 1., 5);
//         frame2.push_square(pos, 1., 21);
//         frame3.push_square(pos, 1., 2);
//
//         let mut raw_data = Vec::new();
//         raw_data.extend_from_slice(frame1.bytes());
//         raw_data.extend_from_slice(frame2.bytes());
//         raw_data.extend_from_slice(frame3.bytes());
//
//         let (mut rx, tx) = pipe().unwrap();
//
//         {
//             let mut writer = Writer::new(tx);
//
//             writer.write(&frame1).unwrap();
//             writer.write(&frame2).unwrap();
//             writer.write(&frame3).unwrap();
//         }
//         // Drop of writer should wait for data to be written
//
//         let mut rx_data = Vec::new();
//         rx.read_to_end(&mut rx_data).unwrap();
//
//         assert!(raw_data == rx_data);
//     }
// }

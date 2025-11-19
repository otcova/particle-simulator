pub use crate::particle::*;
pub use crate::reader::*;
pub use crate::writer::*;

mod particle;
mod reader;
mod writer;

#[cfg(test)]
mod tests {
    use super::*;
    use std::{io, time::Duration};

    #[test]
    fn test_write_read() {
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

        let (rx, tx) = io::pipe().unwrap();

        let reader = Reader::new(rx);

        {
            let mut writer = Writer::new(tx);

            writer.write(&frame1).unwrap();
            writer.write(&frame2).unwrap();
            writer.write(&frame3).unwrap();
        }

        // Give time for the reader thread to start up
        std::thread::sleep(Duration::from_millis(100));

        assert!(reader.read() == Ok(Some(frame1)));
        assert!(reader.read() == Ok(Some(frame2)));
        assert!(reader.read() == Ok(Some(frame3)));
        assert!(reader.read() == Ok(None));
    }
}

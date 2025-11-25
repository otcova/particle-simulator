use std::io::pipe;

use particle_io::{Frame, Reader, Writer};

#[derive(Default)]
pub struct Backend {
    reader: Option<Reader>,
    writer: Option<Writer>,

    pub reader_details: String,
    pub writer_details: String,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum ConnectionState {
    Disconnected,
    Connected,
}

impl Backend {
    pub fn new() -> Backend {
        let mut backend = Backend::default();
        backend.open_itself();
        backend
    }

    pub fn close_connection(&mut self) {
        self.reader = None;
        self.writer = None;

        self.reader_details = "Connection closed".into();
        self.writer_details = "Connection closed".into();
    }

    pub fn open_backend_files(&mut self) {
        let read_path = "./backend_out.bin".into();
        let write_path = "./backend_in.bin".into();

        match Writer::open_file(&write_path) {
            Ok(writer) => {
                self.writer = Some(writer);
                self.writer_details = write_path;
            }
            Err(error) => {
                self.writer = None;
                self.writer_details = error;
            }
        }

        match Reader::open_file(&read_path) {
            Ok(reader) => {
                self.reader = Some(reader);
                self.reader_details = read_path;
            }
            Err(error) => {
                self.reader = None;
                self.reader_details = error;
            }
        }
    }

    pub fn open_tcp(&mut self) {
        let addr = "0.0.0.0:53123";

        match particle_io::new_tcp_server(addr) {
            Ok((reader, writer)) => {
                self.reader = Some(reader);
                self.writer = Some(writer);
                self.reader_details = format!("{} tcp", addr);
                self.writer_details = format!("{} tcp", addr);
            }
            Err(error) => {
                self.reader = None;
                self.writer = None;
                self.reader_details = error.clone();
                self.writer_details = error;
            }
        }
    }

    pub fn open_itself(&mut self) {
        match pipe() {
            Ok((rx, tx)) => {
                self.reader = Some(Reader::new(rx));
                self.writer = Some(Writer::new(tx));
                self.reader_details = "Receiving from itself".into();
                self.writer_details = "Sending to itself".into();
            }
            Err(error) => {
                self.reader = None;
                self.writer = None;
                self.reader_details = error.to_string();
                self.writer_details = error.to_string();
            }
        }
    }

    pub fn reader_connected(&self) -> bool {
        self.reader.is_some()
    }

    pub fn writer_connected(&self) -> bool {
        self.writer.is_some()
    }

    pub fn reader_state(&self) -> ConnectionState {
        if self.reader_connected() {
            ConnectionState::Connected
        } else {
            ConnectionState::Disconnected
        }
    }

    pub fn writer_state(&self) -> ConnectionState {
        if self.writer_connected() {
            ConnectionState::Connected
        } else {
            ConnectionState::Disconnected
        }
    }

    pub fn read(&mut self) -> Option<Frame> {
        let Some(reader) = &self.reader else {
            return None;
        };

        match reader.read() {
            Ok(frame) => frame,
            Err(_) => {
                self.reader = None;
                self.reader_details = "Error".into();
                None
            }
        }
    }

    pub fn write(&mut self, frame: &Frame) {
        let Some(writer) = &mut self.writer else {
            return;
        };

        if let Err(error) = writer.write(frame) {
            self.writer = None;
            self.writer_details = error;
        }
    }
}

use std::{
    borrow::Cow,
    collections::VecDeque,
    io::ErrorKind,
    net::{SocketAddr, TcpListener, TcpStream},
};

use particle_io::{Frame, Reader, TcpClient, Writer};

pub struct Backend {
    reader: Option<Reader>,
    writer: Option<Writer>,

    pub reader_details: String,
    pub writer_details: String,

    tcp_server: Result<TcpListener, String>,

    // In case of disconnected backend, it writes/reads the frames
    // from this queue.
    loopback_queue: VecDeque<Frame>,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum ConnectionState {
    Disconnected,
    Connected,
}

impl Backend {
    pub fn new() -> Backend {
        Backend {
            reader: None,
            writer: None,
            reader_details: "".into(),
            writer_details: "".into(),
            tcp_server: match TcpListener::bind("0.0.0.0:53123") {
                Ok(tcp_server) => {
                    if let Err(error) = tcp_server.set_nonblocking(true) {
                        Err(error.to_string())
                    } else {
                        Ok(tcp_server)
                    }
                }
                Err(e) => Err(e.to_string()),
            },
            loopback_queue: VecDeque::new(),
        }
    }

    pub fn tcp_server_status(&self) -> Cow<'_, str> {
        match &self.tcp_server {
            Ok(server) => server
                .local_addr()
                .map(|addr| addr.to_string())
                .unwrap_or_else(|e| e.to_string())
                .into(),
            Err(error) => error.into(),
        }
    }

    pub fn close_connection(&mut self) {
        self.reader = None;
        self.writer = None;

        self.reader_details = "Connection closed".into();
        self.writer_details = "Connection closed".into();
    }

    pub fn open_backend_files(&mut self) {
        self.loopback_queue.clear();

        let read_path = "./backend_out.bin".into();
        let write_path = "./backend_in.bin".into();

        match Writer::open_file(&write_path) {
            Ok(writer) => {
                self.writer = Some(writer);
                self.writer_details = write_path;
            }
            Err(error) if error.kind() == ErrorKind::NotFound => {
                self.writer = None;
                self.writer_details = format!("File {:?} not found", write_path);
            }
            Err(error) => {
                self.writer = None;
                self.writer_details = error.to_string();
            }
        }

        match Reader::open_file(&read_path) {
            Ok(reader) => {
                self.reader = Some(reader);
                self.reader_details = read_path;
            }
            Err(error) if error.kind() == ErrorKind::NotFound => {
                self.reader = None;
                self.reader_details = format!("File {:?} not found", read_path);
            }
            Err(error) => {
                self.reader = None;
                self.reader_details = error.to_string();
            }
        }
    }

    fn open_tcp(&mut self, stream: TcpStream, backend_addr: SocketAddr) {
        self.loopback_queue.clear();

        match stream.try_clone() {
            Ok(stream2) => {
                self.reader = Some(Reader::new(TcpClient(stream)));
                self.writer = Some(Writer::new(TcpClient(stream2)));
                self.reader_details = format!("{} tcp", backend_addr);
                self.writer_details = format!("{} tcp", backend_addr);
            }
            Err(error) => {
                self.reader = None;
                self.writer = None;
                self.reader_details = format!("{}", error);
                self.writer_details = format!("{}", error);
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

    fn try_accept_tcp_connection(&mut self) {
        if let Ok(server) = &self.tcp_server {
            let Ok((stream, addr)) = server.accept() else {
                return;
            };

            self.open_tcp(stream, addr);
        }
    }

    pub fn read(&mut self) -> Option<Frame> {
        let Some(reader) = &self.reader else {
            self.try_accept_tcp_connection();
            return self.loopback_queue.pop_front();
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
            if self.reader_connected() {
                self.loopback_queue.push_back(frame.clone());
            }
            return;
        };

        if let Err(error) = writer.write(frame) {
            self.writer = None;
            self.writer_details = error;
        }
    }
}

use crate::*;
use std::{
    io::{Error, ErrorKind},
    net::{TcpListener, TcpStream, ToSocketAddrs},
    sync::mpsc,
};

struct TcpReader {
    connection: Option<TcpStream>,
    listener: TcpListener,
    tx: mpsc::Sender<TcpStream>,
}

struct TcpWriter {
    connection: Option<TcpStream>,
    rx: mpsc::Receiver<TcpStream>,
}

impl std::io::Read for TcpReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        loop {
            if let Some(stream) = &mut self.connection {
                match stream.read(buf) {
                    Ok(0) => {
                        self.connection = None;
                    }
                    Ok(n) => {
                        return Ok(n);
                    }
                    Err(error) if error.kind() == ErrorKind::Interrupted => {
                        eprintln!("{}", error);
                        return Err(error);
                    }
                    Err(error) => {
                        eprintln!("{}", error);
                        self.connection = None;
                    }
                }
            }

            for _ in 0..4 {
                match self.listener.accept() {
                    Ok((stream, addr)) => {
                        println!("Client: {}", addr);
                        if self.tx.send(stream.try_clone()?).is_err() {
                            return Ok(0);
                        }
                        self.connection = Some(stream);
                        break;
                    }
                    Err(error) => eprintln!("{}", error),
                }
            }
        }
    }
}

impl std::io::Write for TcpWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        loop {
            if let Some(stream) = &mut self.connection {
                match stream.write(buf) {
                    Ok(n) => return Ok(n),
                    Err(error) if error.kind() == ErrorKind::Interrupted => return Err(error),
                    Err(error) => {
                        eprintln!("{}", error);
                        self.connection = None;
                    }
                }
            }

            match self.rx.recv() {
                Ok(stream) => self.connection = Some(stream),
                Err(_) => return Err(Error::new(ErrorKind::BrokenPipe, "Tcp server closed")),
            }
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        if let Some(stream) = &mut self.connection {
            return stream.flush();
        }
        Ok(())
    }
}

struct TcpClient(TcpStream);

impl std::io::Read for TcpClient {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.0.read(buf)
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> std::io::Result<()> {
        self.0.read_exact(buf)
    }
}

impl std::io::Write for TcpClient {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.write(buf)
    }
    fn write_all(&mut self, buf: &[u8]) -> std::io::Result<()> {
        self.0.write_all(buf)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.0.flush()
    }
}

impl Drop for TcpClient {
    fn drop(&mut self) {
        let _ = self.0.shutdown(std::net::Shutdown::Both);
        println!("Shutdown");
    }
}

pub fn new_tcp_server<A: ToSocketAddrs>(addr: A) -> Result<(Reader, Writer), String> {
    let listener = TcpListener::bind(addr).map_err(|e| e.to_string())?;
    let (tx, rx) = mpsc::channel();
    Ok((
        Reader::new(TcpReader {
            connection: None,
            listener,
            tx,
        }),
        Writer::new(TcpWriter {
            connection: None,
            rx,
        }),
    ))
}

pub fn new_tcp_client<A: ToSocketAddrs>(addr: A) -> Result<(Reader, Writer), String> {
    let client = TcpStream::connect(addr).map_err(|e| e.to_string())?;
    let client2 = client.try_clone().map_err(|e| e.to_string())?;
    Ok((Reader::new(client), Writer::new(TcpClient(client2))))
}

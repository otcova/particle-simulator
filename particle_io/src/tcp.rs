use crate::*;
use std::{
    io::ErrorKind,
    net::{TcpStream, ToSocketAddrs},
};

pub struct TcpClient(pub TcpStream);

impl std::io::Read for TcpClient {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self.0.read(buf) {
            Ok(0) => Err(std::io::Error::new(
                ErrorKind::ConnectionAborted,
                "Tcp connection closed",
            )),
            Ok(n) => Ok(n),
            Err(err) => Err(err),
        }
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
    }
}

pub fn new_tcp_client<A: ToSocketAddrs>(addr: A) -> Result<(Reader, Writer), String> {
    let client = TcpStream::connect(addr).map_err(|e| e.to_string())?;
    let client2 = client.try_clone().map_err(|e| e.to_string())?;
    Ok((
        Reader::new(TcpClient(client)),
        Writer::new(TcpClient(client2)),
    ))
}

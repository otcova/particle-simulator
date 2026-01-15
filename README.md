# Particle Simulator

Source code of the particle simulator.

### Particle Editor

To build the editor [cargo]("https://doc.rust-lang.org/cargo/getting-started/installation.html") is needed,
then one can directly build and run with:
```bash
cargo run --release
```

(First build takes about ~4 minutes)

### Cuda Backend

The following will
1. Compile the rust library
(therefore [cargo]("https://doc.rust-lang.org/cargo/getting-started/installation.html") is needed),
2. Compile & run the simulator. If it fails to compile this one might need to set `CUDA_HOME` manually inside the Makefile.

```bash
cd cuda_simulator
make run
```

Another alternative is to use `build`. This will only try to build the cuda backend,
but for it to work the library will need to be previously compiled correctly.

```bash
cd cuda_simulator
make build
```



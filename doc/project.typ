#set document(title: [Particle Simulator])
#title()

#show link: underline

= Abstract/Intro
[U]
== What is this project about.
In this project we've implemented a pretty simple particle simulator that uses the GPU in order to be able to simulate a lot more particles by using the unmatched parallelism of this type of devices.

== What have we done.
The project could be divided in two big blocks. On the one hand, we have the relevant part to this subject, the simulator that runs on the gpu (or cpu, more on that later). On the other hand, since it's a particle simulator, half the fun is being able to actually *see* the simulation, and for that purpose we've implemented a visualizer/editor that is responsible to both send what to simulate to the simulator and to display the results. This part of the project has been written in #link("https://rust-lang.org/")[Rust].

== What have we seen. TODO:
Well, since particle simulation is a pretty good application in terms of parallelization, running the simulation in the GPU is a lot faster that on the CPU.

= Experiments

== 1. First Working Version [U]
=== Algorithm part
As a first version, we implemented the simplest algorithm possible. For each particle, we will iterate all other particles and calculate the forces that each pair of particles apply to each other. This algorithm is really simple to implement (a simple loop throught the particle list for each particle) and is also easy to parallelize, as each thread can take care of n_particles / n_threads.

The issue with this algorithm is that it's cost is O(n²). While it's okay at simulating a few hundred particles, as you increase the number of particles it starts to struggle quite a bit.

=== General program workings

The overview of how the project works is:
The editor (gui) sends to the "backend" (simulator) a "frame", which contains metadata and the list of particles. The backend will then grab this frame and run the simulation on it.

The backend will, for each frame, do a few iterations over the particles calculating the forces and updating the position. After finishing, it will send the resulting frame to the editor (so it can be displayed). Then it will check if the editor has sent a new frame/scene to simulate, and will then proceed to use the new ones. If not, it continues running the simulation on the previous information.

All the information that the algorithm needs to calculate the force applied to the particle, as well as how many iterations to perform before sending the frame to the editor, is defined in the metadata of the frames and can be adjusted to change the simulation in the editor.

- Explain naive compact implementation with context
- TODO: screenshot (or gif) solid, liquid & gass

== 2. Better Integration [O]
- implemented Leapfrog integration
- upgrade from exploding in 100ns to not doing so ever
- TODO: image of integration by squares & Euler's and Leapfrog integration energy conserving.

== 3. Better datastructure: Buckets [O]
- Explain algotrithm & the needed move step to reallocate the particles from the buckets.
- $O(n^2)$ to $O(n * B^2)$ where B is a constant chosen by us.

== 4. Wall formula [O]
- Less agressive wall formula to prevent acceleration two an immovabol object
- Allows bigger time step x2

== 5. Warp utilization [U]
- Explain how our initial buckets implementation distributes workload on warps.
- Explain our proposed alternatives that introduce a tradeof of memory locality vs warp distribution.
- Results?

== 6. Formula Exponent [O]
- Using other particle parameters for the Mie potential allows us to use bigger time steps
- By reducing exponent, the rigidity of the interactions decreeses, and therefore the maximum forces and accelerations do to. This allows us to advance time in bigger steps, but it also fundamentally changes the simulation. Without the rigidity, solids can not form. Using a very relaxed formula (like an exponent of 2 intead of 14) is coser to what games (and films simulations) use to simulate water.

= How to Run
== Backend setup

The editor and the backend communicate using tcp. Unfortunately, we've not come with a reliable way to auto update the ip that the backend tries to "speak to" and has to be changed manually. The IP to adjust can be found inside the file `frontend.hpp`, in the function `init_tcp()`. By default it will atempt to connect to the own computer, so if the computer has a nvidia GPU, you can just run the simulator on the same computer without issues.

But, if the simulator has to be run in another computer that has a GPU (cough boada cough), it should be as simple as changing the ip inside the `new_tcp_client(&reader, &writer, "0.0.0.0:53123")` by the one that the editor will be running in. If for some reason the editor suddenly stops receiving frames, the ip may have changed (definitively not speaking from experience).

If the ip has to be changed, after doing so go to `project_root/cuda_simulator` and execute make.

== Editor setup

The editor does not need setup.

== Running everything
=== General usage

First, you must have the editor running before the simulator. This is as simple as going into `project_root/build` and running `./particle_editor`. Then, to start the the backend, run `./particle_simulator`.

TODO: Program explanation
- Simple lattice example with explenation

=== Boada specific (non interactive GPU)

For the particle editor, exactly the same. For the simulator, instead of directly running it, we have to use `squeue`. For this, we can use the script `job.sh`. Same as in the rest of the subject, we just have to `squeue job.sh`.

For the rest, same as in general usage.


= About the Source Code

[U]
- General explanation of the project (editor / simulator division)
- How to obtain the whole source code (github) & the requeremnts to built it all.

== Simulation
[U]
- Explain the structure of the .c
- Show small snippets of kernels
- To be able to see the improvement in usign a gpu over a cpu (and to help identify whether an issue is caused by the code or hardware undefined behaviour) we wanted to make it so we had a kernel for the cpu and another one for the gpu.

    Instead of duplicating the code, cuda has a nice feature that allows us to choose for what type of device a kernel should be compiled. In class we've been using `__global__` to "indicate" to the compiler that that function is a kernel for the gpu, but there are more `__XX..X__` keys that can be used. In our case, each kernel has in the declaration `__host__` (compile it for the CPU) `__device__` (compile it for the device/GPU). When the compiler sees both of this it compiles the function for both devices.

- Explain how we use a stream & show a gpu trace

== Editor
[O]
- Explain how this is outside this TGA subject.
- Features:

== Editor Library
[O]
- Explain how this is outside this TGA subject.
- Bridge between the editor and simulaton
- Has TCP support, pipe support and file support.

= Final overview
[U]
- We would have loved to show more cool speedup graphs between versions, but this is not possible to be done fearly when
  1. The max simulation time is increased from a constant to an infinity (leapfrog integration)
  2. The optimization brings a change from cuadratic to practically linear (buckets implementation)
  3. The optimization sacrifices simulation behaviour or acuracy (wall formula & exponent formula).

[O]
- Other optiomization that could have been done:
  - Using group memory to store forces: Since a pari-particle interaction is simetrical, the resulting force only needs to be computed once. Exploiting this could bring performance benefits (although less than 2x improvement given that the force computation is not the 100% of the work (amdals), the overhead of the sincronization, and other factors).
  - Other warp distributions.
  - Using another more elavorate datastructure like a space partition tree instead of fixed constant size buckets. This would remove the simulation box size limit, but it would probably not improve work balancing since we still need to compute all the par interactions of nearby particles (not only the closest one, or the one that is intersection tha particle like in a rigitbody simulation).

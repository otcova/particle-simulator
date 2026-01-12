#set heading(numbering: "1.")
#show heading: set block(above: 2em, below: 1em)
#set page(numbering: "1.")
#show heading.where(level: 3): set heading(outlined: false, numbering: none)

#let code(body, caption: []) = figure(caption: [#caption #v(1em)], supplement: [Snippet])[
  #v(1em)
  #rect(body, stroke: (y: 0.5pt), radius: 5pt, inset: (x: 15pt, y: 10pt))
]



#align(center + horizon)[
  #text(size: 30pt, weight: "bold")[Particle Simulator]

  #v(2em)
  #text(size: 14pt)[
    Otger Comas Valls\
    Uriel Camí Fernández
  ]
]

#pagebreak()
#outline()
#pagebreak()

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

== Leapfrog Integration


=== What is the Integration in our Simulator?
In the context of a particle simulator, integration is the mathematical engine that drives the simulation forward in time.
At every single time step, we calculate the forces acting on every particle using the Mie Potential @mie_potential. The integrator takes these forces and update the position ($x$) and velocity ($v$) of each particle for the next fraction of a second (the time step, $Delta t$).
Without a robust integrator, the simulation is just a static snapshot of forces; the integrator is what makes the particles "move."

=== Euler vs Leapfrog
From Instability to RobustnessOur initial implementation relied on Euler integration (a first-order method). While simple to implement, it suffered from severe energy drift.

- *Previous State (Euler):* The simulation would "explode" (particles gaining infinite energy and flying off) within 100 ns, even when using extremely small time steps of $10^(-16) s$. The error accumulation was too rapid for a stable simulation.
- *Current State (Leapfrog):* By switching to Leapfrog integration (a second-order symplectic method), we achieved total stability. We have successfully tested simulations for durations of at least $10^(-6) s$ using larger time steps of $10^(-14) s$ without any "explosion" or significant energy drift.

@leapfrog_vs_euler illustrates a classic test case of $N$ bodies orbiting a point source mass.
The visualization is perfect to understand that the leapfrog integration is not absent of error, but it's likely to oscilate in over and under compensating producin an asimptotically behaviour.

#figure(
  image("leapfrog.png", width: 70%),
  caption: [
    Comparison of Euler's and Leapfrog integration energy\
    conserving properties for N bodies orbiting a point source mass. @leapfrog_integration
  ],
)<leapfrog_vs_euler>

== Space aware datastructure

=== The Naive Approach
Our initial implementation was designed for simplicity to establish a baseline. In this first version, we stored all particles in a single, unstructured linear list. To calculate the forces for any given particle, we had to iterate through the entire list to compute the interaction with every other particle. While easy to implement (@list_force_computation), this approach has a time complexity of $O(N^2)$. As the number of particles $N$ increased, the computational cost grew quadratically, making the simulation unviable for large-scale systems even on powerful hardware.




#code(
  ```c
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  force = {0, 0};
  for (int j = 0; j < particle_count; ++j)
    force += mie_force(particles[i].pos, particles[j].pos);
  particles[i].apply_force(force);
  ```,
  caption: [Naive force computation $O(N^2)$],
)<list_force_computation>

=== Spatial Partitioning with Buckets

To overcome the $O(N^2)$ limit, we implemented a space-aware data structure. The core idea is simple: particles only interact significantly with those that are nearby. Therefore, we do not need to compute the force for all the pairs of particles that are far apart (@buckets_force_computation).

We divided the simulation space into a uniform grid of squares. Instead of a single global list, we now maintain a specific list of particles for each square.

- *Contiguous Memory Layout:* To maintain cache efficiency on the GPU, we do not use dynamic lists. Instead, each square is allocated a fixed "bucket" in a single large contiguous memory buffer.

- *Bucket Capacity:* Experiments showed that a capacity of 16 particles per bucket was sufficient to hold the local density of particles in our simulation.

- *Unordered Data:* Inside a bucket, particles are not sorted. This reduces overhead, as the order within the square does not affect the physics calculation.

#code(
  ```c
  // Bucket lists are stored contiguously with the following layout:
  // Particle bucket[NUM_BUCKETS][BUCKET_CAPACITY];
  #define BUCKET_CAPACITY 16

  int global_i = blockIdx.x * blockDim.x + threadIdx.x;

  int bucket_i = global_i / BUCKET_CAPACITY;
  int i = global_i % BUCKET_CAPACITY;

  force = {0, 0};
  for (<particles "j" of the adjacent buckets "bucket_j" and itelf>)
    if (i != j || bucket_i != bucket_j)
      force += mie_force(buckets[bucket_i][i].pos, buckets[bucket_j][j].pos);
  buckets[bucket_i][i].apply_force(force);
  ```,
  caption: [Force computation with buckets],
)<buckets_force_computation>

=== Achieving Linear Complexity

By organizing particles into buckets, we drastically reduced the search space for force calculations. For any particle in a given bucket, we only need to iterate through the particles in its own bucket and the 8 neighboring buckets.This transformation changes the complexity from Quadratic to Linear:

#align(center, grid(
  columns: 2,
  gutter: 20pt,
  align(horizon, $O(N dot (16 dot 9 - 1)) approx O(N)$),
  align(left)[
    *$N$:* Total number of particles\
    *$16$:* The maximum capacity of a bucket\
    *$9$:* The number of buckets checked (1 center + 8 neighbors)\
    *$1$:* Excluding the particle itself
  ],
))

// TODO: image space division
// TODO: image 9 adjacent squares

=== Why a Grid and not a Tree?

While Quadtrees or k-d trees are common for spatial partitioning, we chose a uniform grid for two specific reasons:

- *Bounded Simulation:* Our simulation takes place in a compact, bounded space. Trees excel at sparse, unbounded data, but for a dense box of particles, a grid provides faster $O(1)$ access to neighbors without traversing tree nodes.

- *Range vs. Nearest Neighbor:* We do not need to find the single closest particle, we need all particles within a specific interaction radius. By adjusting the square size to match this radius, the grid naturally provides the exact set of candidates needed.


=== Updates & Maintenance
A spatial data structure introduces a new maintenance cost: as particles move, they cross grid lines and must be moved from one bucket to another. We optimized this in two ways:
- *Lazy Updates:* Since particles move relatively slowly compared to the simulation time step ($Delta t$), it is unnecessary to reorganize the memory buffer every single frame. We take advantage of this by only performing a full structure update every $approx 32$ steps (ajustable via a parameter). This amortizes the cost of memory operations over many compute cycles.
- *Parallel "Pull" Updates:* To update the structure, we assign one CUDA thread to each bucket. This thread iterates over the adjacent buckets to check for particles that have drifted into the current bucket's territory (see @buckets_realloc). This parallel "pull" method ensures that the buckets remain consistent with the particle positions with minimal divergence. It may not be the most efficient algorithm, specially due to the small amount of threads created. But since its not the critical path (only executing about 32 times less than a normal simulation step) its not very relevant for performance.



#code(
  ```c
  int bucket_i = blockIdx.x * blockDim.x + threadIdx.x;
  int i = 0;

  // Copy particles inside bucket_i in "buckets_updated[bucket_i]"
  for (<particles "j" of the adjacent buckets "bucket_j" and itelf>){
    Particle p = buckets[bucket_j][j];
    if (<p.pos inside bucket_i>) {
      if (i >= BUCKET_CAPACITY) {
        // Error: Exceading BUCKET_CAPACITY. The simulation needs a bigger grid.
        // This will create more buckets but each will cover a smaller in area.
      }
      buckets_updated[bucket_i][i] = p;
      i += 1;
    }
  }

  // Fill the remaining slots of "buckets_updated[bucket_i]" with a sentinel
  while (i < BUCKET_CAPACITY) {
    buckets_updated[bucket_i][i] = null;
    i += 1;
  }
  ```,
  caption: [Reallocating particles in their buckets],
)<buckets_realloc>



== Wall formula


=== The Boundary Problem

In particle simulations, there are two standard ways to handle the edges of the simulation box:

1. *Periodic Boundaries:* Particles loop back to the opposite side (like Pac-Man).
2. *Reflective Boundaries:* Particles bounce off the walls.

For our simulation, we chose Reflective Boundaries to simulate a confined container. However, implementing this proved challenging. We iterated through three distinct methods to find a solution that did not compromise our time step ($Delta t$).



=== Attempt 1: Velocity Inversion (Hard Bounce)


Our first approach was the standard kinematic solution: if a particle's coordinate crosses a boundary, we simply invert the sign of its velocity vector on that axis ($v = -v$).

While this works for single particles, it caused major instability with dense groups (e.g., a solid lattice structure). When a cluster of particles hit the wall, the instantaneous change in velocity created a shockwave. The simulation would "explode" due to the non-physical discontinuity of the energy. To make this work, we had to reduce the simulation time step ($Delta t$) significantly, which ruined our performance.


=== Attempt 2:"Ghost" Particle Walls

We realized that in the real world, walls are simply made of other atoms. To replicate this, we placed "ghost" particles at the four boundaries of the simulation box.
Instead of a hard coordinate check, the particles inside the box now interact with these wall particles using the standard force calculation (@wall_forces).

#code(
  ```c
  forces += mie_force(particle_pos, {0,              particle_pos.y});
  forces += mie_force(particle_pos, {box_width,      particle_pos.y});
  forces += mie_force(particle_pos, {particle_pos.x, 0             });
  forces += mie_force(particle_pos, {particle_pos.x, box_height    });
  ```,
  caption: [Computation of wall forces],
)<wall_forces>

The particles now experience a gradual push as they approach the wall, rather than an instant snap. This successfully removed the "explosion" issues seen in the first attempt.

However, there is still a limit. Since the wall particles are immovable (infinite mass), an incoming particle must come to a full stop and reverse solely on its own. In a normal collision between two free particles, both move aside, distributing the energy. Because the wall refuses to move, the collision is much harsher. Experiments showed this forced us to use a $Delta t$ roughly half of what is safe for normal inter-particle collisions.


=== Attempt 3: Repulsive-Only Potential

The inefficiency in the attempt 2 specially came from the nature of the inter-particle force, which contains two components:

- *Repulsive Force:* Pushes particles apart when too close.
- *Attractive Force:* Pulls particles together when slightly separated.

Using the full potential meant that as a particle approached the wall, it was first attracted to it—accelerating it "like crazy" right before the impact. This high-speed impact with an immubable particle required a smaller time step to resolve accurately.

In this attemt 3 we modified the formula to apply only the repulsive component of the potential. And this change is not a total compromise from speed and reality. Since we can assume that the walls are made from a different material, the interaction will not have the same attraction/repulsive parameters.

This way particles are no longer sucked into the wall before hitting it. They drift towards the wall and are gently pushed back by the repulsive field. This change allowed us to double the $Delta t$ compared to Attempt 2, restoring our simulation speed while maintaining perfect stability.

// TODO: graph of mie force vs repulsive only mie force


== Warp utilization [U]<warp-utilization>
- Explain how our initial buckets implementation distributes workload on warps.
- Explain our proposed alternatives that introduce a tradeof of memory locality vs warp distribution.
- Results?

== Force Formula: Stiffness vs. Speed

Beyond algorithmic improvements (like Leapfrog or Spatial Partitioning), another way to increase the simulation speed is to modify the fundamental laws governing the particles.

To allow for even larger time steps (higher $Delta t$), we experimented with altering the exponent in the inter-particle force formula (such as the repulsive term in the Mie potential).

=== Reducing the Exponent
The exponent in the force formula essentially dictates the "stiffness" or "hardness" of the particles.
- *High Exponent (e.g., 14):* This creates a "hard" shell. The repulsive force spikes drastically over a very short distance. This mimics real solid matter where atoms cannot overlap. However, these massive spikes in acceleration require extremely tiny time steps to resolve accurately, or the simulation explodes.

- *Low Exponent (e.g., 2):* This creates a "soft" shell. The force changes gradually as particles approach each other. Because the maximum acceleration is much lower, the integrator can handle much larger jumps in time ($Delta t$) without instability.

=== The Trade-off
While reducing the exponent allows the simulation to run significantly faster, it fundamentally changes the nature of the matter being simulated.

Our experiments showed that there is a critical limit to this optimization.

- *With High Stiffness:* We can simulate Solids. Particles lock into place to form rigid lattice structures if their velocity (temperature) is low enough.

- *With Low Stiffness:* We lose the ability to form solids. Using a very relaxed formula (e.g., an exponent of 2), the particles become "squishy." Instead of stacking into a structure, they slide past one another or compress like a soft gas or fluid.

Therefore, this optimization is not a "free" like the others. it is a parameter choice. If we need to simulate a crystal, we must accept the cost of a higher exponent and smaller $Delta t$. If we only need to simulate a gas or a soft liquid, we can lower the exponent to gain performance.

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

== Future Optimizations & Alternative Approache
While our current implementation achieves linear complexity and stability, there are other advanced optimization strategies that we considered or identified as potential future improvements.

*1. Exploiting Newton's Third Law (Symmetrical Forces)*

Physical forces are symmetrical: the force particle $A$ exerts on $B$ is equal and opposite to the force $B$ exerts on $A$ ($F_(i j) = -F_(j i)$).

Theoretically, we could halve the number of force calculations by computing the interaction once and applying the result to both particles simultaneously. This would likely involve using Group Shared Memory to accumulate forces locally before writing to global memory.

Why we didn't prioritize it? The actual speedup would be significantly less than $2times$:
- *Amdahl's Law:* Force calculation is not 100% of the execution time.
- *Synchronization Overhead:* Writing to two different memory locations simultaneously requires thread synchronization, which introduces significant overhead that can negate the computational savings.


*2. Optimizing Warp Occupancy*

Our current "Thread-per-Slot/Particle" approach simplifies the logic but suffers from Warp distribution.

Since we allocate fixed-size buckets (capacity 16) but many buckets may only contain a few particles, threads assigned to empty slots remain idle. In CUDA, if one thread in a warp is active and the others are not, the hardware must still execute the instructions for the empty threads, leading to wasted cycles.

We already tryied to sove this in @warp-utilization, but we did not cover it all. Other solution could be:
- A more complex scheduler that compacts the active particles into a dense list before processing. This would ensure that every warp is fully saturated with work, maximizing the GPU's throughput.
- A smarter distribution of particles to minimize mixing empty and full slots into the same warp.
- Other work subdivision rather than one particle per thread.

*3. Sparse Spatial Structures (Trees)*

We currently use a fixed grid, which imposes a hard limit on the simulation box size.
We could implement more sparse datastructures (like a Quadtree). This could even allow the simulation to be unbounded, particles could travel infinitely far without hitting an "end of the world."

However, while trees are more flexible, they are not necessarily faster for this specific type of physics.
Unlike a rigid body simulation that only checks for collisions (intersection), our simulation calculates forces based on a potential field. We need to find all particles within a radius, not just the single closest one. Traversing a tree to find all neighbors in a radius is computationally heavier than iterating over a known fixed grid of buckets.



#bibliography("bibliography.yml")

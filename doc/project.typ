#set heading(numbering: "1.")
#show heading: set block(above: 2em, below: 1em)
#set page(numbering: "1")
#show heading.where(level: 3): set heading(outlined: false, numbering: none)
#show link: underline

#let code(body, caption: []) = figure(caption: [#caption #v(1em)], supplement: [Snippet])[
  #v(1em)
  #rect(body, stroke: (y: 0.5pt), radius: 5pt, inset: (x: 15pt, y: 10pt))
]


#import "@preview/lilaq:0.5.0" as lq

#let s = 3.4
#let n = 12
#let m = 6
#let C = (n / (n - m)) * calc.pow(n / m, m / (n - m))
#let p = 0.00000118

#let x = lq.linspace(2, 10, num: 200)
#let y = x.map(x => C * p * (m * calc.pow(s / x, m) - n * calc.pow(s / x, n)) / x)


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

= Abstract/Intro

== What is this project about

This project focuses on the implementation and optimization of a simple particle simulator using CUDA to leverage GPU parallelism.
We experiment on implementing different integration methods for better simulation stability, implement a space-aware data structure to achieve linear-complexity and more.

== What have we done
The project could be divided in two big blocks. On the one hand, we have the relevant part to this subject, the simulator that runs on the gpu (or cpu, more on that later). On the other hand, since it's a particle simulator, half the fun is being able to actually *see* the simulation, and for that purpose we've implemented a visualizer/editor that is responsible to both send what to simulate to the simulator and to display the results. This part of the project has been written in #link("https://rust-lang.org/")[Rust].

We've been iteratively improving the algorithm of the simulator to allow us (mainly) to simulate more particles, and to make it more stable, since the first versions were quite prone to suddenly stop working.

= Simulation Basics

This section explains some relevant theory about our particle simulator. It give enough context to better comprehend the following sections.

== Mie Potential


At the core of our simulator lies the Mie Potential. It is the fundamental law that describes how every particle interacts with its neighbors. It dictates the behavior of the particles that we simulate.

The Mie potential is a generalized form of the famous Lennard-Jones potential. It calculates the force ($F$) acting on a particle based on its distance ($r$) from another particle. This force is a delicate balance between two opposing components:

- *The Repulsive Force:* A short-range, powerful force that pushes particles apart. This prevents them from overlapping and effectively simulates the "collision" of solid matter.

- *The Attractive Force:* A weaker, longer-range (but still only a few angstroms) force that pulls particles together. This simulates the Van der Waals forces that allow matter to condense into liquids and solids.


This interplay of forces allows our simulator to model different states of matter naturally, without explicit programming for "solid" or "liquid" behaviors.

- *Solids:* When particles have low kinetic energy (low temperature), the attractive force traps them in the "potential well" of their neighbors. They vibrate but cannot escape, locking into a lattice structure.

- *Melting:* If we heat the system (increase particle velocity), the kinetic energy of the particles eventually overcomes the weak attractive force. They break free from the lattice and begin to flow, simulating the phase transition from solid to liquid.


#v(1em)
#figure(caption: [Force formula derived from the Mie Potential])[
  #grid(
    columns: 2,
    gutter: 20pt,
    align(horizon)[
      $F(r) = C dot epsilon dot [ m(sigma/r)^m - n(sigma/r)^n]$
    ],
    align(left)[
      *$F(r)$:* Force relative to the distance "r" of the particles\
      *$sigma$, $epsilon$, $n$, $m$:* Particle parameters. Different values describe the\ interaction between different particles of the periodic table.
    ],
  )
  #line(length: 70%, stroke: 0.5pt)
]
#v(1em)


=== Simulation Limitations


To understand why particle simulations are so prone to "exploding," we can look at the shape of this force function.

@force_plot shows the force between two particles as a function of distance. The tip above the X-axis ($"Force" > 0$) represents the attractive region where particles sit comfortably when forming a solid. The curve decreasing sharply to the left is the repulsive region.


#v(1em)
#figure(
  caption: [
    Plot example of the force between a pair of particles.\
  ],
)[
  #lq.diagram(
    width: 100%,
    ylabel: [Force ($N$)],
    xlabel: [Particle Distance ($angstrom$)],
    ylim: (-.000001, .000001),
    xlim: (2, 10),
    lq.plot(
      x,
      y,
      label: "Complete Mie Force",
      smooth: true,
      mark: none,
    ),
  )
  #v(1em)
]<force_plot>
#v(1em)


When we zoom out the Y-axis by a factor of 100 (@force_wall_plot), the true nature of the simulation challenge becomes visible. The repulsive force does not just increase; it creates a near-vertical "wall". If a particle moves just half an angstrom ($0.5 angstrom$) too close to another, the repulsive force doesn't just double, it spikes by orders of magnitude. This extreme stiffness is why we need such minuscule time steps ($Delta t$). If the time step is too large, a particle might accidentally step deep into this "wall" in a single frame. The resulting calculated force would be astronomically high, causing the particle to be ejected at a physically impossible speed—instantly destroying the simulation.



#figure(
  caption: [
    100x zoom out of the force plot (@force_plot).\
  ],
)[
  #lq.diagram(
    width: 100%,
    ylabel: [Force ($N$)],
    xlabel: [Particle Distance ($angstrom$)],
    ylim: (-.0001, .00001),
    xlim: (2, 10),
    lq.plot(
      x,
      y,
      label: "Complete Mie Force",
      smooth: true,
      mark: none,
    ),
  )
  #v(1em)
]<force_wall_plot>

= Experiments

== First Working Version
=== Algorithm part
As a first version, we implemented the simplest algorithm possible. For each particle, we will iterate all other particles and calculate the forces that each pair of particles apply to each other. This algorithm is really simple to implement (a simple loop throught the particle list for each particle) and is also easy to parallelize, as each thread can take care of n_particles / n_threads.

The issue with this algorithm is that it's cost is O(n²). While it's okay at simulating a few hundred particles, as you increase the number of particles it starts to struggle quite a bit.

=== General program workings

The overview of how the project works is:
The editor (gui) sends to the "backend" (simulator) a "frame", which contains metadata and the list of particles. The backend will then grab this frame and run the simulation on it.

The backend will, for each frame, do a few iterations over the particles calculating the forces and updating the position and speed. After finishing, it will send the resulting frame to the editor (so it can be displayed). Then it will check if the editor has sent a new frame/scene to simulate, and will then proceed to use the new one. If not, it continues running the simulation on the previous information.

All the information that the algorithm needs to calculate the force applied to the particles, as well as how many iterations to perform before sending the frame to the editor, is defined in the metadata of the frames and can be adjusted in the editor to change the behaviour of the simulation.

Below are a few captures of performed simulations:

#grid(
  columns: 2,
  figure(
    image("Solid.gif", width: 90%),
    caption: [Simulating a solid],
  ),
  [
    #figure(
      image("Liquid.gif", width: 90%),
      caption: [Solid with imperfections\ (some holes and layer shifts)],
    )
    #v(30pt)
  ],

  figure(
    image("Liquid.png", width: 90%),
    caption: [Liquid blob with some evaporated particles],
  ),
  figure(
    image("Gas.gif", width: 90%),
    caption: [High-Pressure gas],
  ),
)

#v(5em)



== Leapfrog Integration


=== What is the Integration in our Simulator?
In the context of a particle simulator, integration is the mathematical engine that drives the simulation forward in time.
At every single time step, we calculate the forces acting on every particle using the Mie Potential @mie_potential. The integrator takes these forces and update the position ($x$) and velocity ($v$) of each particle for the next fraction of a second (the time step, $Delta t$).
Without a robust integrator, the simulation is just a static snapshot of forces; the integrator is what makes the particles "move."

=== Euler vs Leapfrog
Our initial implementation relied on Euler integration (a first-order method). While simple to implement, it suffered from severe energy drift.

- *Previous State (Euler):* The simulation would "explode" (particles gaining infinite energy and flying off) within $100p s (= 10^(-10) s)$, even when using extremely small time steps of $1 f s (= 10^(-15) s)$. The error accumulation was too rapid for a stable simulation.
- *Current State (Leapfrog):* By switching to Leapfrog integration (a second-order symplectic method), we achieved "total stability". We have successfully tested simulations for durations of more than $10 n s (= 10^(-8) s)$ using larger time steps of $10 f s (= 10^(-14) s)$ without any "explosion" or significant energy drift. Even so, this does not mean the simulation is perfect, just that the energy drift does not have a trend to increase or decrease indefinitely.

@leapfrog_vs_euler illustrates a classic test case of $N$ bodies orbiting a point source mass.
The visualization is perfect to understand that the leapfrog integration is not absent of error, but it's more likely to oscillate producing a more or less stable simulation.

#v(1em)
#figure(
  image("leapfrog.png", width: 80%),
  caption: [
    Comparison of Euler's and Leapfrog integration *energy\ drift over time*
    for N bodies orbiting a point source mass. @leapfrog_integration
  ],
)<leapfrog_vs_euler>
#v(1em)

This optimization/improvement might be the most relevant for two reasons:
1. If we were to pick a speedup, it could be $infinity$ since the previous version lasted $~100p s$ and the new one lasts indefinitely.
2. The code change is absurdly small since we only need to change the used a formula as also shown in @leapfrog_vs_euler.

== Space aware datastructure

=== The Naive Approach
Our initial implementation was designed for simplicity to establish a baseline. In this first version, we stored all particles in a single, unstructured linear list. To calculate the forces for any given particle, we had to iterate through the entire list to compute the interaction with every other particle. While easy to implement (@list_force_computation), this approach has a time complexity of $O(N^2)$. As the number of particles $N$ increased, the computational cost grew quadratically, making the simulation unviable for large-scale systems even on powerful hardware.

#code(
  ```c
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  force = {0, 0};
  for (int j = 0; j < particle_count; ++j)
    if (i != j)
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
  align(horizon)[
    $O(N dot (16 dot 9 - 1)) approx O(N)$
  ],
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
  for (<particles "j" of the adjacent buckets "bucket_j" and itself>){
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
  caption: [Relocating particles in their buckets],
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

#v(1em)
#align(center, grid(
  columns: 2,
  gutter: 40pt,
  figure(caption: [Complete Mie Force])[
    $F(r) = C dot epsilon dot [ m(sigma/r)^m - n(sigma/r)^n]$
  ],
  figure(caption: [Only Repulsive Mie Force])[
    $F(r) = C dot epsilon dot [ - n(sigma/r)^n ]$
  ],
))






#figure(
  caption: [
    Representative plot of the forces between the simulation\
    particles, and the force between th particles and the walls
  ],
)[
  #lq.diagram(
    width: 100%,
    ylabel: [Force ($N$)],
    xlabel: [Particle Distance ($angstrom$)],
    ylim: (-.000001, .000001),
    xlim: (2, 10),
    lq.plot(
      x,
      y,
      label: "Complete Mie Force",
      smooth: true,
      mark: none,
    ),
    lq.plot(
      x,
      x => C * p * (-n * calc.pow(s / x, n)) / x,
      label: "Only repulsive Mie Force",
      smooth: true,
      mark: none,
    ),
  )
  #v(1em)
]


// TODO: graph of mie force vs repulsive only mie force


== Warp utilization<warp-utilization>
- The warp utilization (for the buckets version) is not that great, since the particles are usually concentrated in a few buckets, which means that some of the warps will have a lot of work, while others will not have to do anything.

In order to try to remedy this, we tried to change it so that the block 0 would take care of the first particle inside each bucket, block 1 the second particle, block 2 the third.... This way, since all blocks would work on all the buckets, instead of some of them doing all the work every one of them would do more or less equal work.

From our results, it did not appear to make a difference. We assume that due to the memory access pattern being worse, the potential performance improvement from having better warp utilization did not pay off.

== Force Formula: Stiffness vs. Speed

Beyond algorithmic improvements (like Leapfrog or Spatial Partitioning), another way to increase the simulation speed is to modify the fundamental laws governing the particles.

To allow for even larger time steps (higher $Delta t$), we experimented with altering the exponent in the inter-particle force formula (such as the repulsive term in the Mie potential).

=== Reducing the Exponent
The exponent in the force formula essentially dictates the "stiffness" or "hardness" of the particles.
- *High Exponent ($approx 14$):* This creates a "hard" shell. The repulsive force spikes drastically over a very short distance. This mimics real solid matter where atoms cannot overlap. However, these massive spikes in acceleration require extremely tiny time steps to resolve accurately, or the simulation explodes.

- *Low Exponent ($approx 2$):* This creates a "soft" shell. The force changes gradually as particles approach each other. Because the maximum acceleration is much lower, the integrator can handle much larger jumps in time ($Delta t$) without instability.

=== The Trade-off
While reducing the exponent allows the simulation to run significantly faster, it fundamentally changes the nature of the matter being simulated.

Our experiments showed that there is a critical limit to this optimization.

- *With High Stiffness:* We can simulate Solids. Particles lock into place to form rigid lattice structures if their velocity (temperature) is low enough.

- *With Low Stiffness:* We lose the ability to form solids. Using a very relaxed formula (e.g., an exponent of 2), the particles become "squishy." Instead of stacking into a structure, they slide past one another or compress like a soft gas or fluid.

Therefore, this optimization is not a "free" like the others. it is a parameter choice. If we need to simulate a crystal, we must accept the cost of a higher exponent and smaller $Delta t$. If we only need to simulate a gas or a soft liquid, we can lower the exponent to gain performance.

= How to Run
== Backend setup<backend_setup>

The editor and the backend communicate using tcp. Unfortunately, we've not come with a reliable way to auto update the ip that the backend tries to "speak to" and has to be changed manually. The IP to adjust can be found inside the file `frontend.hpp` (full path: _project_root/cuda_simulator/src/lib/frontend.hpp _), in the function `init_tcp()`. By default it will atempt to connect to the own computer, so if the computer has a nvidia GPU, you can just run the simulator on the same computer without issues.

But, if the simulator has to be run in another computer that has a GPU (cough boada cough), it should be as simple as changing the ip inside the `new_tcp_client(&reader, &writer, "0.0.0.0:53123")` by the one that the editor will be running in. If for some reason the editor suddenly stops receiving frames, the ip may have changed (definitively not speaking from experience).

#code(
  ```c
  void init_tcp() {
      // Change this IP ---------------------------------↴
      is_connected = new_tcp_client(&reader, &writer, "0.0.0.0:53123");
  }
  ```,
  caption: "init_tcp function",
)

If the ip has to be changed, after doing so go to _project_root/cuda_simulator _ and execute make.

== Editor setup

The editor does not need setup, as the executable will be provided.

== Running everything
=== General usage

First, you must have the editor running before the simulator. This is as simple as going into _project_root/build/ _ and running
```bash
./particle_editor &
```
Then, to start the backend, run
```bash
./particle_simulator
```

The editor has two main parts. The Left Panel, where there are mainly controls and information about the simulation / the visualizer. And the Right Panel, which has at the upper part the visualizer of the simulation and below that some controls for moving throught the playback of the simulation.

In the Left Panel we can find subsections, which are:
- Backend: Mainly useful to se wheter we are connected to the backend or not.

- Editor: Here we can control what to send to the backend to simulate, as well as the ability to create/edit/delete presets. For a first test, you can try to send a Lattice Preset (after adjusting the parameters if wanted) with either the `Hexagonal Square` button (this will create a structure that should resemble a solid, with the particles staying more or less stable) or the `Square` button (this is a bit more random on what happens. Maybe it colapses into a solid similar to a hexagonal square, maybe it explodes).

  There is also the user presets, which is where we can draw or own shapes. If we click in either the second button of an already created preset we will edit that preset, and if we click the `New preset from:` we will create a new preset that depending of what we choose on the right Menu will have either no particles or will contain the particles of the current frame.\
  Once we've entered "edit mode", the visualizer will become a canvas to edit. The bottom panel instead of having buttons to control the playback, will have tools we can use to edit the frame (brush to add particles, eraser to remove particles, speedometer to modify the speed of particles, and broom to clear the canvas). Each tool has a bunch of options, which hopefully are self explanatory enough to understand.

  Finally, there is the `clear and send next` button, which clears the simulation "history" when sending the next shape; and the Edit & Resend current, which can be used to quickly change the current simulation frame.

- Parameters: Here we have mostly parameters which control the frames metadata. The most useful ones are:
  - step delta time: How much time passes between each iteration of the simulation (the bigger the faster the simulation will go. Making it to big will most likely explode the simulation)
  - Steps per frame: Changes how many iterations on the simulation we do before sending it to the editor.
  - Box width: Width of the box. Might not play nice with the current simulation if used in interactive mode.
  - Box height: Same as width, but height.
  - Data structure: Whether to simulate using buckets or compact array. Useful to see the improvement of buckets over naive approach.
  - Device: whether to run on GPU or CPU. Will probably break the simulation if changed in interactive mode.
  - GPU threads/block: Allows to adjust how many threads we give to each block.
  - Particle X: Changes parameters about how the particles interact with each other. Unfortunately we did not implement this fully and only particle 0 matters, as that is what the simulator uses.

- Stats: Shows stats about the simulation and current frame.

- Timeline: Mainly useful to see when you have to clear the simulation so it doesn't crash the computer due to missing ram.

- GUI: Allows changing parameters about the gui. Useful/interesting ones are:
  - Max speed for Color: Will change how fast the particles must be going before their color changes noticeably.
  - Min Particle Size: So the particles are easier to see. Useful if the box is too big.

On the Visualizer, the main elements are the actual window where the frames will be seen, and the bottom panel which can be showing different things as explained before. The playback buttons are pretty intuitive, so we will not be explaining them.

What we will be explaining is that there is an *interactive mode*, which gets activated once we are on the current frame of the Timeline and the `loop playback` option is not on (and we are not editing a frame). In this mode, if we adjust the parameters in the `Stats` section of the left panel it will be sent to the simulator in real time. Apart from being nice to be able to see the changes that you are doing affecting the simulation in real time (and also how depending on what you adjust the simulation instantly explodes), in this mode you also can affect the simulation in a more direct way. If you left click on the visualizer, the particles inside the circle that will show up will be pushed away. To control the size of the circle, the `Cursor Size` on the `Stats` section can be changed. Note that making it to big is a really good way to break the simulation.

Finally, note that there are the following keyboard shortcuts:
- ESC: Close the program.
- F11: Go fullScreen.
- Space: Play/Stop the playback.
- Left arrow: Go back in the playback.
- Right arrow: Go forward in the playback (or go to the beggining if we are at the last frame).
- C: Cleat the simulation timeline.
- L: sends a Lattice preset.
- D: Disconnects from the backend.


=== Boada specific (non interactive GPU)

First and foremost, connect to boada using `-X` as ssh option, else the editor... will look a lot less interesting :)

For the particle editor, exactly the same. For the simulator, instead of directly running it, we have to use `squeue`. For this, we can use the script `job.sh`. Equal as in the rest of the subject, we just have to
```bash
    squeue job.sh
```

For the rest, same as in general usage.


= About the Source Code

As mentioned before, the project has two big blocks, that being the editor and the simulator. There is also a more secondary block (but necessary), which is the library that allows both of them to communicate.

The whole source code can be found and downloaded at #link("https://github.com/otcova/particle-simulator"). In order to compile it, a newish version of #link("https://doc.rust-lang.org/cargo/getting-started/installation.html")[cargo] (rust main package manager) will be needed.

The editor can be compiled by going into _project_root/particle_editor/ _ and running ```bash
    cargo build --release
```
(note: the first time will take up a while, because it has to basically install and compile 400\~ish rust libraries).

In a similar fashion, going into _project_root/particle_io_ and running the same command will compile the library necessary to communicate the editor and the simulator.

Finally going into _project_root/cuda_simulator_ and doing `make` will compile the simulator (refer to @backend_setup)

== Simulation
- The source of the simulator is in _project_root/cuda_simulator_, which has the following structure.
  - `cuda_simulator`: Root folder of the simulator.
    - `Makefile`: Makefile that compiles the source code of the simulator using nvcc. It can compile into "release" mode and "debug" mode.

  - `build`: Folder where the compiled code will go.

    - `src`: Folder where the source code of the simulator is.
      - `cuda_simulator.cu`: Contains the main loop of the simulator.

      - `kernel.cuh`: Has some ```c #define``` that the rest of the code uses, as well as functions to abstract the code of the cuda version versus the CPU version.

    - `kernel_bucket.cuh`: Has the kernels and the calls to them for the bucket version.

    - `kernel_compact.cuh`: Has the kernels and the calls to them for the compact/naive version.

    - `particle.cuh`: Functions to calculate things related with the particles (forces, velocities, positions).

    - `lib`: Folder which contains helper code:
      - `frontend.hpp`: used as an interface between the simulator and the library that communicates the simulator and the editor.

      - `thread_pool.hpp`: custom thread_pool implementation to give the CPU a more equal figthing ground against the GPU.

      - `log.hpp`: used for debugging purposes.

- Below are the three kernels where calculations happen. As mentioned before, the compact kernel is a lot simpler than the other two. The two `step kernels` are missing the code for adding the walls force, the cursor interaction and the code to apply the force to the position and velocity of the particle at the end of the kernel. In addition, the two `bucket kernels` are missing the code that checks which buckets they should work with (so the buckets next to walls don't try to acces things that they shouldn't).

#code(
  ```c
      ...
      for (uint32_t j = 0; j < particle_count; ++j) {
          if (j == i) continue;

          float2 r = f_dist(src[i], src[j], frame);
          force += params.f2_force(r);
      }
      ...
  ```,
  caption: [compact step kernel],
)

#code(
  ```c
      ...
      for (int32_t y = y_min; y <= y_max; ++y) {
          for (int32_t x = x_min; x <= x_max; ++x) {
              uint32_t bucket_j = ((x + bucket_x) + (y + bucket_y) * BUCKETS_Y) * BUCKET_CAPACITY;

              for (uint32_t jj = 0; jj < BUCKET_CAPACITY; ++jj) {
                  uint32_t j = jj + bucket_j;
                  if (j == i || src[j].ty < 0) continue;

                  float2 r = f_dist(src[i], src[j], frame);
                  force += params.f2_force(r);
              }
          }
      }
      ...
  ```,
  caption: [bucket step kernel],
)

#code(
  ```c
      ...
      for (int32_t y = y_min; y <= y_max; ++y) {
          for (int32_t x = x_min; x <= x_max; ++x) {
              uint32_t bucket_j = ((x + bucket_x) + (y + bucket_y) * BUCKETS_Y) * BUCKET_CAPACITY;

              for (uint32_t jj = 0; jj < BUCKET_CAPACITY; ++jj) {
                  uint32_t j = jj + bucket_j;
                  if (src[j].ty < 0) continue;

                  if (src[j].x >> (32 - BUCKETS_X_LOG2) != bucket_x ||
                      src[j].y >> (32 - BUCKETS_Y_LOG2) != bucket_y) continue;


                  dst[bucket_i*BUCKET_CAPACITY + i++] = src[j];
                  if (i == BUCKET_CAPACITY) return;
              }
          }
      }
      ...
  ```,
  caption: [bucket move kernel],
)

- To be able to see the improvement in usign a gpu over a cpu (and to help identify whether an issue is caused by the code or hardware undefined behaviour) we wanted to make it so we had a kernel for the cpu and another one for the gpu.

  Instead of duplicating the code, cuda has a nice feature that allows us to choose for what type of device a kernel should be compiled. In class we've been using `__global__` to "indicate" to the compiler that that function is a kernel for the gpu, but there are more `__XX..X__` keys that can be used. In our case, each kernel has in the declaration `__host__` (compile it for the CPU) and `__device__` (compile it for the device/GPU). When the compiler sees both of this it compiles the function for both devices.

- In order to improve performance, we use two streams to separate the calculations from the memory transfers. This in theory allows us to start calculating the next frame while we transfer the previous one to the host and then send it to the editor. On the GPU that Uriel has (RTX3050) this does not appear to work, as can be seen on the trace below.

#figure(
  image("Trace.png"),
  caption: [Example Trace],
)

== Editor

=== Scope and Purpose
The Editor serves as the graphical interface for our project, providing tools to visualize the simulation in real-time and interactively feed data (initial conditions, particle types) into the engine. However, the Editor is strictly a visualization and control tool, it does not perform the physics calculations. As such, it falls outside the core scope of this subject, which focuses on the GPU accelerated simulator implementation. Therefore, we will only outline its high-level architecture.


== Technology Stack
Since the editor lays outside the subject we were not bound by C/C++. We chose Rust for its development, primarily to leverage existing personal projects and its modern ecosystem. With ecosystem we primarily mean the following libraries:

- *egui:* Used for the user interface. It provides an immediate mode GUI that is highly responsive, allowing us to reuse UI code from previous personal work.

- *winit:* Library for window creation and event handling, ensuring the application is cross-platform and OS-independent. Even so, we only directly give support for linux, the editor might work in other systems.

- *wgpu:* A modern graphics API wrapper (WebGPU standard) used for rendering. While OpenGL was an option, wgpu offers better integration with Rust and we reused some code from existing personal projects. Even so our rendering code is simple enough: we utilize a single instanced draw call to render all particles at once.

== Editor Library

=== The Bridge
To connect the Rust-based Editor with the C++/CUDA Simulator, we created a dedicated intermediate library. This "Editor Library" acts as the communication bridge, ensuring data flows seamlessly between the visualization layer and the physics engine.

=== Shared Logic
Since both the Editor and the Simulator require the same communication protocols, implementing the logic twice (once in Rust, once in C++) would be redundant and error-prone. Instead, we centralized this logic:

- *Implementation:* The core networking and serialization logic was written in Rust.
- *Compilation:* This Rust code is compiled as a library that exposes a C-compatible ABI (Application Binary Interface).
- *Usage:* This allows the exact same compiled binary to be linked into the C++ Simulator and imported into the Rust Editor, guaranteeing identical behavior on both ends.

=== Communication Channels
The library is designed to be agnostic about the transport method, supporting multiple ways to exchange data:

- *TCP Sockets:* The primary method uses the Rust standard library's TCP implementation to allow the Editor and Simulator to run as separate processes.

- *Pipes & Files:* The architecture also supports communication via named pipes or files for local data transfer. Initially implemented as a fallback measure in case we encountered problems establishing a TCP connection with the server compute nodes.


= Final overview
We would have loved to show more cool speedup numbers and graphs between code versions, but this is not possible to be done fairly when
1. The max simulation time is increased from a constant to an infinity (leapfrog integration)
2. The optimization brings a change from cuadratic to practically linear (buckets implementation)
3. The optimization sacrifices simulation behaviour or accuracy (wall formula & exponent formula).


== Future Optimizations & Alternative Approaches
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

We already tried to sove this in @warp-utilization, but we did not cover it all. Other solution could be:
- A more complex scheduler that compacts the active particles into a dense list before processing. This would ensure that every warp is fully saturated with work, maximizing the GPU's throughput.
- A smarter distribution of particles to minimize mixing empty and full slots into the same warp.
- Other work subdivision rather than one particle per thread.

*3. Sparse Spatial Structures (Trees)*

We currently use a fixed grid, which imposes a hard limit on the simulation box size.
We could implement more sparse datastructures (like a Quadtree). This could even allow the simulation to be unbounded, particles could travel infinitely far without hitting an "end of the world."

However, while trees are more flexible, they are not necessarily faster for this specific type of physics.
Unlike a rigid body simulation that only checks for collisions (intersection), our simulation calculates forces based on a potential field. We need to find all particles within a radius, not just the single closest one. Traversing a tree to find all neighbors in a radius is computationally heavier than iterating over a known fixed grid of buckets.



#bibliography("bibliography.yml")

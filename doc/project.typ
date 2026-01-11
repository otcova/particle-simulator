#set heading(numbering: "1.")
#show heading: set block(above: 2em, below: 1em)
#set page(numbering: "1.")
#show heading.where(level: 3): set heading(outlined: false, numbering: none)

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
[U]
- What is this project about.
- What have we done.
- What have we seen.

= Experiments

== First Working Version [U]
- Explain naive compact implementation with context
- TODO: screenshot (or gif) solid, liquid & gass

== Leapfrog Integration [O]


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


== Better datastructure: Buckets [O]
- Explain algotrithm & the needed move step to reallocate the particles from the buckets.
- $O(n^2)$ to $O(n * B^2)$ where B is a constant chosen by us.

== Wall formula [O]
- Less agressive wall formula to prevent acceleration two an immovabol object
- Allows bigger time step x2

== Warp utilization [U]
- Explain how our initial buckets implementation distributes workload on warps.
- Explain our proposed alternatives that introduce a tradeof of memory locality vs warp distribution.
- Results?

== Formula Exponent [O]
- Using other particle parameters for the Mie potential allows us to use bigger time steps
- By reducing exponent, the rigidity of the interactions decreeses, and therefore the maximum forces and accelerations do to. This allows us to advance time in bigger steps, but it also fundamentally changes the simulation. Without the rigidity, solids can not form. Using a very relaxed formula (like an exponent of 2 intead of 14) is coser to what games (and films simulations) use to simulate water.

= How to Run
[U]
- Editor & Backend Execution
- Simple lattice example with explenation

= About the Source Code

[U]
- General explanation of the project (editor / simulator division)
- How to obtain the whole source code (github) & the requeremnts to built it all.

== Simulation
[U]
- Explain the structure of the .c
- Show small snippets of kernels
- (Extra) Explain how we can define functions to be callable by also the cpu, and therefore use the cpu to run code.

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

- Explain how using parallelism we can increase the amount of particles simulated, but not the speed of such simulation since simulation steps are sequential, so methods like this one that allow for a larger time step with better results are better than duplicating the hardware.

[O]
- Other optiomization that could have been done:
  - Using group memory to store forces: Since a pari-particle interaction is simetrical, the resulting force only needs to be computed once. Exploiting this could bring performance benefits (although less than 2x improvement given that the force computation is not the 100% of the work (amdals), the overhead of the sincronization, and other factors).
  - Other warp distributions.
  - Using another more elavorate datastructure like a space partition tree instead of fixed constant size buckets. This would remove the simulation box size limit, but it would probably not improve work balancing since we still need to compute all the par interactions of nearby particles (not only the closest one, or the one that is intersection tha particle like in a rigitbody simulation).

#bibliography("bibliography.yml")

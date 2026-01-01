#set document(title: [Particle Simulator])
#title()

= Abstract/Intro
[U]
- What is this project about.
- What have we done.
- What have we seen.

= Experiments

== 1. First Working Version [U]
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

[O]
- Other optiomization that could have been done:
  - Using group memory to store forces: Since a pari-particle interaction is simetrical, the resulting force only needs to be computed once. Exploiting this could bring performance benefits (although less than 2x improvement given that the force computation is not the 100% of the work (amdals), the overhead of the sincronization, and other factors).
  - Other warp distributions.
  - Using another more elavorate datastructure like a space partition tree instead of fixed constant size buckets. This would remove the simulation box size limit, but it would probably not improve work balancing since we still need to compute all the par interactions of nearby particles (not only the closest one, or the one that is intersection tha particle like in a rigitbody simulation).

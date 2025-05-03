# Implementing Embedded DSLs for CUDA in Python

Hello everyone. My name is Andy Terrel. I’ve been contributing bugs to Python libraries for nearly 20 years but haven’t been involved much with the core language development. My domain of expertise lies more in scientific and data algorithms, especially when it comes to scaling across multiple machines. You might know me as a Director of NumFOCUS, basically the PSF for scientific codes written (mostly) in Python. I currently moonlight at NVIDIA helping teams make their technology useful to the Python world.

## Python, the great enabler

To me, Python has been an amazing language and community. The language has been focused on making computational thinking accessible to non-experts from the start. It’s implementation is kept clean and accessible. Features have been implemented with an eye towards simplicity. In a world of non stop change the simplicity and straight-forward nature of the Python language has made a good technology to use for innovation in other domains, e.g. novel numerical algorithms or machine learning.

From the start of my career working in Python, I have been working on embedding languages within Python. First with the FEniCS projects that used Python to model numerical algorithms called finite elements and generate C++ simulations. These simulations were used to develop mechanical systems, and one I’m most proud of was the team who used it to find way to mitigate strokes by studying the blood flow in the brain. Next  with SymPy for building symbolic algorithms for discovering new numerical algorithms. This has produced a huge number of scientific discoveries from all  walks of science. Over 1800 citations as of this writing, but whose keeping track of that =D

These techniques were picked up by Theano, possibly the first deep learning system, which later frameworks replicated, especially Tensorflow and PyTorch. These frameworks needed to be able to differentiate algorithms to implement the back propagation algorithm. Personally this is a story very familiar to me as when trying to apply a novel fluid model to a toy problem, I realized there would be over a million integrals to analyze and thus I started working on SymPy. Theano first stated using SymPy but quickly built it’s own internal dalect. 

## GPUs, the accelerator

At the same time these languages are evolving a new technological trend in computing hardware emerged, the use of GPUs for general purpose computing. Pythonistas have been building on GPU for over 15 years. My first time speaking at PyCon was a tutorial on GPUs in 2012. GPUs are effectively thousands of small cpus with threads coordinated and executed in groups with various shared resources, e.g. registers, and global memory. Because of the nature of the hardware, GPUs have a much different programming model than CPUs, we call it SIMT (Share Instructions Multiple Threads). The art of making programs (called kernels) that run on the GPU has largely been left to a minority of engineers. Deep learning frameworks used GPUs from the very early days, mostly utilizing libraries built by vendor.

Over the last several years as deep learning models have grown very large, researchers have discovered that mixing various numerical types in the calculation will allow them to use more parameters within the same budget. Now we have a situation that has caused a challenge for the community. It is no longer a library binding problem, that can be solved by enumerating a number of functions and pre compiling them, but a language problem, where algorithms need to be expressed over various types of numbers and enumeration would be more costly than just in time compilation. Thus we have seen numerous domain specific languages evolve to address this issue. Unfortunately, in the rush to innovate, these languages are implementing different semantics and are often not interoperable. They have not learned the lesson the core Python team has taught us for many years, that collaboration and ruthless focus on simplicity leads to sustainable software.

## Python GPU DSLs, the odd couple

Now we are seeing languages evolve very quickly with array languages, graph languages, block languages, and so on. Just to show a list of Python DSLs targeting GPUs includes here are 16 in a cursory search, but there are certainly many more as I can personally attest that almost every company I’ve worked in has a few more internally.  

1. High-Level Array Libraries (NumPy-like)
    - CuPy
    - JAX
    - PyTorch (autograd + CUDA tensors)
2. Kernel-Level DSLs
    - Numba-CUDA
    - Triton
    - Warp
    - PyCUDA
    - CuPy Raw Kernel
    - Kernel Tuner
    - Pallas
3. Model Graph Compilers
    - TorchScript
    - TensorFlow XLA
    - TorchInductor
4. Task & Dataflow DSLs
    - DaCe
    - Parla
5. Experimental / Domain-specific DSLs
    - Luxon
    - Aetherling



Each one of these languages has problems working with the ecosystem in a few key ways, for example typing rules, calling conventions, and describing shared resources.  To highlight this we have been working together at NVIDIA to come up with a guide on how to build DSLs targeting GPUs. We certainly believe that helping people use these devices better will help technologists create more novel solutions just as Python has been a part of for decades. Some of these problems are certainly an exercise for each language implementer, but we do think there are things that the core language can do to help the   language implementor. Okay I’ve used most of my 5 minutes but let me push through just the highest level of these recommendations. Also one caveat, none of these DSLs I mentioned actually implement all these recommendations, so yeah this is the N+1 specification problem =P

### Typing rules

First let’s tackle typing rules. Python was built on C and ctypes remains the core of the language. NumPy came along and added a typing system that corresponds to more of the types that scientific users need. But today new types are being added to the mix at a blazing speed. Some are just smaller versions of known types, e.g. integers of 2 or 4 bits. Others are reinterpretation of the bits in an old type, e.g. brain floats. Thus we have at least three type systems in use at the moment: numba (numpy), jax.ml_types and PyTorch types. 

These type systems usually are discovered by a user when their code silently starts returning bad results. This is because we have trained our users to expect the widest type to always win, but with new mixed typing algorithms that is not always the performant choice. In some areas the jit function can throw an error, but in many cases it needs to make a decision. 

Perhaps there is a simple typing flag or common way for a user to express their intentions.


### Calling conventions


### Shared resources

## Core Language Community, the great collaborator

Thank you for listening to me ramble a bit today. Coming to  talk here was a bit of a journey, a bit like a wizard visiting a different school, and I don’t really have a clear ask from the community. It seems to me that the core language community is very good at defining abstractions that have worked across the various Python communities. I look forward to talking with you at the conference and perhaps even working together at the sprints. Please feel free to ping me on any of the various places you might find me, or just email me and invite me to your space.
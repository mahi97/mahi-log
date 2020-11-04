---
layout: post
comments: true
title: "Neural Turing Machine"
date: 2020-10-30 12:00:00
tags: meta-learning deep-learning machine-learning memory-augmented
---

> Neural Turing Machine (NTM) is one of the first practices for utilizing an external memory to improve neural networks performance, It aims to learn how to use a memory bank and act like a computer program in order to achieve a more generalize solution and with far less samples. In this post we discuss how NTM interact and utilize a memory and show a robust implementation of NTM with PyTorch.


<!--more-->

*Can Neural Network Learn Program?*

Have you ever though having a pen and piece of paper can help you in solving a problem?
Have you ever write a program that can do better only by having more amount of memory?
Having more a piece of paper doesn't make you any smarter but you know how utilize the pen and paper to solve more complex problem faster.

Same story goes for Neural Networks we leave them to solve our problems without any external memory and it leads to an unnatural way of learning which usually is not **Robust**, **Scalable** and needs **Huge** amount of samples.

The Alex Graves,(DeepMind) claims this in his paper on NTM:
> We extend the capabilities of neural networks by coupling them to external memory resources, which they can interact with by attentional processes.

But there's no official implementation of his work, so in this post we discuss how NTM works and also, present a robust implementation of NTM which trained on different tasks and achieve performance near what reported in original paper.

{: class="table-of-content"}
* TOC
{:toc}

## Motivation

When we face complex problems: In one hand we have training of large-scale deep learning models which is not so *natural*, e.g. we need massive amount of data to train specific task with no ability to generalize or deal with noises, no human learn like this.
On the other hand we have good old symbolic AI, which cannot keep up in accuracy and speed with Neural Network, but really understandable and robust.
From the early days of Neural Network, (when they call it connectivisim approach or distributed data processing) symbolic AI scientist raises to critics about it:
  1. Incapable of handling variable sized input
  2. Incapable of “variable-binding” (eg. Mary Spokes to John; we bind Mary to subject role.)

After some years **Recurrent** architecture appears and solved the first critic, but Neural Network lack the ability to bind variable which is one of key requirement of any computer algorithm. The NTM aims to solve this second critics.

## Inspiration

In NTM paper first source of inspiration comes from **Neuroscience and Psychology**; It claims that even humans have a **Working Memory**(e.g. short-term memory storage and rule based manipulation) to handle variable binding and also known as **Rapidly Created Variables**.

The second inspiration comes from **Turing Machine** and **Von Neumann architecture**, it show that even a simple Finite-State Machine with infinity long tape can simulate any algorithm or a Controller can utilize a simple ALU and large enough memory to solve complex problems.
So maybe neural network architecture can also solve complex problem by using simple architecture and memory, instead of a large complex architecture.

## Intuition

This is how a simple **Von Neumann Architecture** look like:
![Von Neumann Architecture]({{ '/assets/post-fig/Von_Neumann_Architecture.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Figure 1. CPU interact with memory to process inputs and generate ourputs . (Image source: Wikipedia)*

By this idea we can assume the NTM as follow:
![Nerual Truing Machine]({{ '/assets/post-fig/ntm_architecture.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Figure 2. a NN as controller interact with memory with help of Read/Write heads like Turing Machine . (Image source: Original Paper)*

## Mathematics
First, we discuss how we can read, write and address a memory in differentiable manner and then we discuss how a Neural Network can generate address and information which suite this memory.

### Memory

The Memory is a matrix with size of $${m \times n}$$ which we show it in time $$t$$ as $$M_t$$
For read in a differentiable manner from memory we define the address as a distribution of rows of matrix and output will be the expected value of address over content of the memory.
the $$w_t(i)$$ is address in the form of:

$$
\sum_i{w_t(i)} = 1\\
0 \le w_t(i) < 1, i \in (0, ..., m-1)
$$  

and we have $$r_t$$ the read vector of memory as:

$$
r_t \leftarrow \sum_i{w_t(i)M_t(i)}
$$

similar to read, we can define the write, with two additional vector for add $$a_t$$ and erase $$e_t$$.

$$
\tilde{M}_t \leftarrow M_{t-1}(i)[1 - w_t(i)e_t],\\
M_t(i) \leftarrow \tilde{M}_t + w_t(i)a_t.
$$

It can as read and write heads use simple attention method to interact with memory.

### Heads

Now we look into more important part: **Addressing!**

  1. Focusing by content:
      The first step to address a memory location is use similarity between memory values and controller output, this approach seems like searching inside memory to find a close match for a query.
      At the time $$t$$ each head produce a key-strength $$\beta$$ and a key-vector $$k_t$$ of length $$m$$. The address (distribution) can be generate as follows:

      $$
        w^c_t(i) \leftarrow \frac{\exp{(\beta_tK[k_t, M_t(i)]}}{\sum_j \exp{(\beta_tK[k_t, M_t(j)])}} \leftarrow Softmax(\beta_tK[k_t, M_t(i)])\\
        K[u,v] = \frac{u.v}{||u|| . ||v||} \leftarrow \textbf{ Cosine Similarity }
      $$

      ![Addressing by Content]({{ '/assets/post-fig/ntm_addr_1.png' | relative_url }})
      {: style="width: 100%;" class="center"}
      *Figure 2. Addressing By Content Similarity . (Image source: Original Paper)*

  2. Interpolation
      In the second step we controller the ability for Interpolate the address between the last address and new address, this Interpolation decide by one gate value $$g_t$$ an it goes as follows:

      $$
        w^g_t \leftarrow g_tw^c_t + (1 - g_t)w_{t-1}.
        $$

      ![Addressing by Content]({{ '/assets/post-fig/ntm_addr_2.png' | relative_url }})
      {: style="width: 100%;" class="center"}
      *Figure 2. Addressing By Content Similarity . (Image source: Original Paper)*

  3. Convolutional Shift
      This step create the ability the shift Interpolated address, this help for iteration over memory cells or get a value before/after the query. each head decide the shift amount by generating a distribution over allowable shift values (i.e. [-2, -1, 0, 1, 2]) as $$s_t$$.

      $$
        w^s_t \leftarrow \sum^{n-1}_{j=0}w^g_t(j)s_t(i - j)
      $$

      ![Addressing by Content]({{ '/assets/post-fig/ntm_addr_3.png' | relative_url }})
      {: style="width: 100%;" class="center"}
      *Figure 2. Addressing By Content Similarity . (Image source: Original Paper)*

  4. Sharpening
      We need a differentiable way to apply the shift so we used the Convolutional Shift in last step, but the Convolutional shift also make the output vector blurry, so reduce this bluriiness, there's a sharpening step after shift and power of sharpening is decided by $$\gamma_t$$.

      $$
        w_t \leftarrow \frac{w^s_t(i)^{\gamma_t}}{\sum_j{w^s_t(j)^{\gamma_t}}}
      $$

      ![Addressing by Content]({{ '/assets/post-fig/ntm_addr_4.png' | relative_url }})
      {: style="width: 100%;" class="center"}
      *Figure 2. Addressing By Content Similarity . (Image source: Original Paper)*

And that's it! the distribution vector $$w_t$$ is generated for each head based on last state of memory and controller output, the system is entirely differentiable and thus end-to-end trainable.

### controller

The controller can be any NN, the paper test how Feed-Forward or RNN perform on different tasks. Beside final results the paper make following statements:
  1. The LSTM version of RNN has internal memory which can act like registers of processor  
  2. LSTM mix information across multiple time-steps so task can be done with less heads
  3. Feed-Forward has better transparency which only relays on external memory


## Implementation

They say you didn't understand it unless you can explain it well,
I believe in our subject you didn't understand it unless you implement it well, and as there was no official implementation for this paper I was eager to do it myself and see what can goes wrong, and as I expected anything that can go wrong did go wrong so I learned few tricks to make it robust and fast.

  1. Here's what I understood first time read the papers, I decide on those activation functions respected to bounds and behavior of variables. seems good, doesn't work!

  ![Implementation 1]({{ '/assets/post-fig/ntm_imp_diag1.png' | relative_url }})
  {: style="width: 70%;" class="center"}
  *Figure 2. Addressing By Content Similarity . (Image source: Original Paper)*

  2. **Moore to Mealy**. First trick is which improved the speed of training and also robustness was changing from moore to mealy which means not only state of controller decide on output but also current values of read heads should be involved.

  ![Implementation 1]({{ '/assets/post-fig/ntm_imp_diag2.png' | relative_url }})
  {: style="width: 70%;" class="center"}
  *Figure 2. Addressing By Content Similarity . (Image source: Original Paper)*


  3. **Split the FC!**. Yes! it actually help a lot and have gain in performance.

  ![Implementation 1]({{ '/assets/post-fig/ntm_imp_diag3.png' | relative_url }})
  {: style="width: 70%;" class="center"}
  *Figure 2. Addressing By Content Similarity . (Image source: Original Paper)*

  4. **Clip output and gradient**. This is very important in robustness of training otherwise you may face g radiant exploit. I tried to clip it in range of $$[-10, 10]$$ and works fine.

  ![Implementation 1]({{ '/assets/post-fig/ntm_imp_diag4.png' | relative_url }})
  {: style="width: 70%;" class="center"}
  *Figure 2. Addressing By Content Similarity . (Image source: Original Paper)*

  5. **Initialization Matters**. The initial values of memory and heads matter a lot in training, there's actually a paper [] on comparing the performance of different Initialization, here's what I found: you should either go with constant Initialization of memory and trainable Initialization for heads, or just initialize everything uniformly random. (I did the second one, but first is recommended).

  ![Implementation 1]({{ '/assets/post-fig/ntm_imp_diag5.png' | relative_url }})
  {: style="width: 70%;" class="center"}
  *Figure 2. Addressing By Content Similarity . (Image source: Original Paper)*

## Experiment & Result

The paper suggest 5 tasks, which I implement and test 3 of them.
  1. Copy Task
  2. Repeated Copy Tasks
  3. Associative Recall
  4. Priority Sort (*Not Implemented*)
  5. Find N-Gram (*Not Implemented*)

Before jumping to plots and graphs, I just want to recall what we expected to see in this results:
  1. Can NTM learn more **Natural**? (e.g. less samples)
  2. Can NTM **Generalize** beyond training range?
  3. How NTM utilize the memory at all?

### Can NTM learn more **Natural**?


### Can NTM **Generalize** beyond training range?


### How NTM utilize the memory at all?

## Conclusion
Neuroscience and computer architecture suggest use of external memory
NTM offers a possible solution to a key criticism of connectionism (variable-binding)
Blurry reads and writes are critical for learning how to use memory
NTMs can outperform and learn more generalizable algorithms than LSTMs
NTM memory access is natural (Is it learning algorithm?)


## Future Works

Any Drawbacks of NTM?
NTM Lacks the support for pointer and data structures
No link between data or ability to backtrack
There’s no way to unallocate memory



Open Questions?
What is the true role of memory in deep learning?
How can memory be traded with learnable parameter?
What is the best way to utilize memory?
How the data is encoded, stored and extracted in the memory?
Can we learn the memory size and structures too?

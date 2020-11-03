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

Same story goes for Neural Networks we leave them to solve our problems without any external memory and it leads to an unnatural way of learning which usually is not *Robust*, *Scalable* and needs *Huge* amount of samples.

The Alex Graves,(DeepMind) claims this in his paper on NTM:
> We extend the capabilities of neural networks by coupling them to external memory resources, which they can interact with by attentional processes.

But there's no official implementation of his work, so in this post we discuss how NTM works and also, present a robust implementation of NTM which trained on diffrent tasks and achieve performance near what reported in original paper.

{: class="table-of-content"}
* TOC
{:toc}

## Motivation

When we face complex problems: In one hand we have training of large-scale deep learning models which is not so *natural*, e.g. we need massive amount of data to train specific task with no ability to generalize or deal with noises, no human learn like this.
On the other hand we have good old symbolic AI, which cannot keep up in accuracy and speed with Neural Network, but really understandable and robust.
From the early days of Neural Network, (when they call it connectivisim approach or distributed data processing) symbolic AI scientist raises to critics about it:
  1. Incapable of handling variable sized input
  2. Incapable of “variable-binding” (eg. Mary Spokes to John; we bind Mary to subject role.)

After some years *Recurrent* architecture appears and solved the first critic, but Neural Network lack the ability to bind variable which is one of key requirement of any computer algorithm. The NTM aims to solve this second critics.

## Inspiration

In NTM paper first source of inspiration comes from **Neuroscience and Psychology**; It claims that even humans have a *Working Memory*(e.g. short-term memory storage and rule based manipulation) to handle variable binding and also known as *Rapidly Created Variables*.

The second inspiration comes from *Turing Machine* and *Von Neumann architecture*, it show that even a simple Finite-State Machine with infinity long tape can simulate any algorithm or a Controller can utilize a simple ALU and large enough memory to solve complex problems.
So maybe neural network architecture can also solve complex problem by using simple architecture and memory, instead of a large complex architecture.

## Intuition

This is how a simple *Von Neumann Architecture* look like:
![Von Neumann Architecture]({{ '/assets/post-fig/Von_Neumann_Architecture.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Figure 1. CPU interact with memory to process inputs and generate ourputs . (Image source: Wikipedia)*

By this idea we can assume the NTM as follow:
This is how a simple *Von Neumann Architecture* look like:
![Nerual Truing Machine]({{ '/assets/post-fig/ntm_architecture.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Figure 2. a NN as controller interact with memory with help of Read/Write heads like Turing Machine . (Image source: Original Paper)*

## Mathematics
First, we discuss how we can read, write and address a memory in differential manner and then we discuss how a Neural Network can generate address and information which suite this memory.

The Memory is a matrix with size of $${m \times n}$$ which we show it in time $$t$$ as $$M_t$$
For read in a differential manner from memory we define the address as a distribution of rows of matrix and output will be the expected value of address over content of the memory.
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

Now we look into more important part: **Addressing!**



## Implementation
## Experiment & Result
## Conclusion
## Future Works

---
layout: post
comments: true
title: "Neural Turing Machine"
date: 2020-10-30 12:00:00
tags: meta-learning deep-learning machine-learning memory-augmented
---


> Neural Architecture Search (NAS) automates network architecture engineering. It aims to learn a network topology that can achieve best performance on a certain task. By dissecting the methods for NAS into three components: search space, search algorithm and child model evolution strategy, this post reviews many interesting ideas for better, faster and more cost-efficient automatic neural architecture search.


<!--more-->


Although most popular and successful model architectures are designed by human experts, it doesn't mean we have explored the entire network architecture space and settled down with the best option. We would have a better chance to find the optimal solution if we adopt a systematic and automatic way of learning high-performance model architectures.




{: class="table-of-content"}
* TOC
{:toc}



## Search Space

The NAS search space defines a set of basic network operations and how operations can be connected to construct valid network architectures.


### Sequential Layer-wise Operations

---
layout: post
comments: true
title: "Killing Ego: Part 2"
date: 2025-11-23 01:00:00
---

> In this part, I want to introduce a mathematical framework on seeing the world without ego and how to optimzie our life and our society.
> This framework closely resemble Markov Decision Process and Reinforcement Learning litreture and allows a both representation and optimization.

<!--more-->


In [Killing Ego Part 1](https://mahi97.github.io/mahi-log/2025/11/19/killing-ego.html),
we discussed the idea that by removing the concept of ego and “who a person is” or “what they have”,
and only focusing on their **actions**, we can more clearly evaluate, justify, and reward people for positive behaviour.

In this framework, we ignore all of someone’s **past actions** and only judge their **current action** given the **current state** they are in.
The state itself is the result of their previous actions and the initial conditions they were born into.
Because of that, the state itself is not “rewardable”; it only **conditions** how we reward their **current action**.

This gives us something that looks almost exactly like a **Markov Decision Process (MDP)** and a **Reinforcement Learning (RL)** optimization problem. Once we phrase life like this, we can steal a lot of beautiful mathematical tools that were built for MDPs and RL.

---

### What is a Markov Decision Process (MDP)?

Very informally, an MDP is a clean way to describe a world where:

1. There is an **agent** (you).
2. There is an **environment** (everything that is not you).
3. Time moves in **steps**: (t = 0, 1, 2, \dots)
4. At each step:

   * The world is in some **state** (s_t).
   * You choose an **action** (a_t).
   * The world responds with a new **state** (s_{t+1}).
   * You **reward** (r_t) yourself in your head, that says how “good” that step was for your chosen goal.

The key assumption is the **Markov property**:

> The future depends on the **present state and action**, not directly on the full detailed past.

If the current state is rich enough and well-designed, then you don’t need the entire history. You can cut the past off and just use the state.

Formally, an MDP is usually defined as a 5‑tuple:

$$
\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, r, \gamma)
$$

where:

* $mathcal{S}$: set of all possible **states**
* $mathcal{A}$: set of all possible **actions**
* $P(s' \mid s, a)$: **transition function** – the probability that the next state will be $s'$ if we are now in state $s$ and take action $a$
* $r(s,a,s')$: **reward function** – the (expected) reward we get when we move from state $s$ to state $s'$ using action $a$
* $\gamma \in [0,1)$: **discount factor**, how much we care about future rewards compared to present ones

At each time step $t$:

1. We see the current state $s_t \in \mathcal{S}$.
2. We choose an action $a_t \in \mathcal{A}$.
3. The environment samples the next state:
   $$
   s_{t+1} \sim P(\cdot \mid s_t, a_t)
   $$
4. We get a reward:
   $$
   r_t = r(s_t, a_t, s_{t+1})
   $$

Mathematically, it’s just some sets and functions. Conceptually, it’s a neat way to talk about **situations**, **choices**, **consequences**, and **how good those consequences are**, without dragging in ego, identity, or stories. Just: state → action → next state → reward.

---

### What is Reinforcement Learning (RL) and how can it optimize the MDP?

If an MDP is the **world**, Reinforcement Learning is the art of **learning how to act** in that world.

In RL, we assume:

* The agent **does not know** all the details of the environment:

  * It doesn’t know the exact transition function $P$.
  * It might not know the reward function precisely either.
* The agent has to **learn** what to do by **interacting**:

  * trying actions,
  * seeing what happens,
  * and adjusting its future behaviour.

The object we want to learn is a **policy**:

$$
\pi(a \mid s)
$$

This is a rule that says: given state $s$, how likely are we to choose action $a$?
A deterministic policy is just “in state $s$, always do action $a$.”

The goal in RL is to find a policy that **maximizes the expected total reward over time**.
Commonly we define the **return** starting at time $t$ as:

$$
G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}
$$

RL then tries to find a policy $\pi$ that maximizes:

$$
\mathbb{E}\big[ G_0 \mid \pi \big]
$$

Two important tools appear here:

* **Value function**
  The value of a state $s$ under a policy $\pi$ is:
  $$
  V^\pi(s) = \mathbb{E}\big[ G_t \mid s_t = s, \pi \big]
  $$
  It answers: *“If I find myself in this state and continue acting according to $\pi$, how good is my life expected to be from now on?”*

* **Action-value function (Q-function)**
  The value of taking action $a$ in state $s$ under policy $\pi$ is:
  $$
  Q^\pi(s,a) = \mathbb{E}\big[ G_t \mid s_t = s, a_t = a, \pi \big]
  $$
  It answers: *“How good is it to do this specific action here, assuming afterward I keep following $\pi$?”*

RL algorithms (Q-learning, policy gradient, actor–critic, etc.) are just different ways to **estimate these values** and then **improve the policy** based on them.

From an ego-less perspective, this is very attractive.
We don’t say “I am a good person” or “You are a bad person”.
We say:

* “This **policy** leads to higher long-term reward.”
* “This kind of **action in this kind of state** is good / bad, regardless of who does it.”

Optimization is now about **improving the policy**, not about proving the goodness of an state, agent, or ego.

---

### How can we represent Life as an MDP + RL?

The goal of the MDP representation is to **separate history from the current state**.
We try to encode the past into a **state representation** that captures all the important effects of the past on the future. Then:

* We judge the **current action** based on the **current state**.
* We do **not** reward or punish the state itself *before* any action happens.

So, for example, we should ignore:

* how someone looks,
* how rich they are,
* what education they have,
* what high achievements they already got.

All of those are part of their **state**.
They may be **interesting**, they may catch our attention, but they are not by themselves something we should reward.

What matters for reward is:

* Given this state, **what action do they choose now?**
* How much **benefit or harm** does that action bring?

A person in an “interesting” state simply has **more potential** to generate impactful actions:

* A rich person has the **potential** to take an action that makes others rich.
* A poor person might not be able to do that, simply because their state doesn’t allow such actions.
* A funny person has the potential to give us a good time.
* A good-looking person has the potential to give us aesthetic or physical pleasure.
* An intellectual mind has the potential to give us ideas that revolutionize our beliefs.

But **potential alone** is not the thing we should reward.
Unless they are willing to **take actions** that benefit others, they should not be rewarded.

So the actual “potential” of a person, in our MDP sense, is:

> The **expected return** of their future actions, given:
>
> * the benefit these actions can create for us (reward),
> * and the **probability** that they will actually take those actions.

Now comes the problem: we do **not** fully observe the true inner state of other people.
We don’t know all their capacities, intentions, and probabilities of different actions.

This is a **Partially Observable MDP (POMDP)**, or in a better sense it is a **Decentralized POMDP (Dec-POMDP)** as we are all trainable agents.

In a Dec-POMDP:

* We don’t see the full state $s$.
* We only get **observations** $o$, which are partial, noisy, and sometimes misleading.
* Decision making now depends heavily on **information gathering** and **beliefs**.

Some information about people is free and easy to validate (public behaviour, past actions).
Other information is:

* costly,
* hidden,
* or outright false.

(See [Everyone Lies](https://mahi97.github.io/mahi-log/2025/11/18/everyone-lies.html))

Because of this, **true information becomes a kind of currency of power**.
The more accurate, filtered information we have (and the better we are at removing lies and noise), the better we can optimize our strategy.

Most lies are cheap to produce:

* Advertising is often free to consume.
* People quickly pretend they’re in a good state and highly likely to reward you if you “invest” in them.

High-quality information, on the other hand, is expensive:

* It may take a long time,
* a lot of money,
* careful attention,
* or some ultimate life “revelation”.

(See [The Final Game](https://mahi97.github.io/mahi-log/2025/11/19/the-final-game.html))

Because we cannot collect **all** information with high quality, we must design a **strategy**:

* how to explore the world,
* how to collect information,
* and how to do it **as cheaply as possible**.

---

### How to collect high-quality information cheaply?

Given our current information, we have to decide:

> **Where** should we acquire more information, and **which** information will actually be useful for our goals?

If we start with **no information at all**, we are almost doomed to accept reality “as presented” – like in *The Truman Show*.
We are easily manipulated by whoever controls the narrative.

But in practice, we always have **some** background knowledge, and we can leverage that to:

* evaluate which sources are more reliable,
* and navigate the information landscape more intelligently.

Imagine moving to a new city or country and knowing no one.
At first, it is hard to trust or validate the information you receive. You rely on:

* small free social clues,
* how people behave,
* how consistent their stories are,

and slowly you try to figure out who is more trustworthy.

At this early stage, what matters is not the **quantity** of information but its **quality**.
With a small but reliable knowledge base, you can then:

* choose which sources to trust,
* and decide which types of information will be most useful.

This is why it is so easy to take advantage of **newcomers and tourists**: they have not yet developed a knowledge base that can filter and validate new information effectively.

However, with careful and cautious moves, we can collect enough reliable information to start playing the game well.

After this initial phase:

1. When someone is in a “good state” (rich, skilled, connected, etc.), they catch our attention.
2. We then seek **additional information**:

   * What is the likelihood they will actually take a beneficial action?
   * How accurate is our observation about their true inner state?

We must be **smart** in two ways:

* to get this information as **cheaply** as possible, and
* to know **which states** are worth exploring at all.

Since we can never know everything for sure, we develop a **probabilistic model** of other people:

* We assign probabilities to their possible actions.
* We assign probabilities to whether the information we received is true or false.

Once we have these beliefs, we can estimate the **expected return** from interacting with different people.
We can then decide **how much to invest** in them.

If we have access to **high-quality information** and we use it well, in the long run we should be able to:

* win overall from our long-term investments,
* even though many individual investments will fail or even harm us.

The important idea:

> We aim to win **in expectation** over many interactions, not to be right every single time.

Our investments should be clear and (when appropriate) transactional, but not in a way that “kills the cat”, meaning not in a way that completely destroys the natural relationship.
Sometimes making transactions too explicit changes the nature of decision-making itself.

However, in **serious and high-stakes situations**, it is foolish to rely only on instincts and vibes.
In those cases we should:

* explicitly define the transaction,
* write contracts,
* and define penalties and rewards for success or failure.

People are continuously seeking investments from each other (time, attention, money, love, support), and we are doing the same.
So it is not strange to want:

* more **transparency**,
* more **clarity**,
* especially when the stakes are high.

Sometimes investments are **two-sided**.
In these cases:

* what we give and what we take are both forms of investment,
* and we do not want a one-sided transaction.

For example, in friendship or romantic relationships:

* we want someone who adds value to our life,
* and who also perceives us as adding value to theirs,
* such that both sides feel they get **more** than they put in.

This is possible because we live in different “currencies” of needs:

* Someone might trade qualities of a woman for counter-parts in a man.
* Friends share **intellect** and **resources**, creating an alliance that makes both stronger.

So broadly there are two cases:

1. **Sharing (mutual investment)**
   where both sides invest and both sides receive.
2. **One-sided transactional investment**
   where we invest in someone mainly for the expected return, without strong mutual emotional ties.

If we can look at all this **mathematically and strategically** over the long term, we can:

* optimize our life given the society we live in,
* and also, in small ways, optimize the society around us given our life goals.

But we must always remember:

* people lie,
* information can be attacked, distorted, or weaponized.

So information is **gold**, the **currency of power**, and we must be very careful:

* about what we believe,
* who we trust,
* and who might be a deceiving agent.

---

### What do I mean by all this?

Life is learned through experience, and anyone can master it without a degree in mathematics.

You do not need to think about life through the lens of MDPs or RL. This post is simply an exercise in stretching the mind, in seeing how far we can push a mathematical view of human behaviour and social interaction.

The real takeaway is something much simpler:

*separate ego from action.*

Most deception in life comes from confusing the two. People present an inflated picture of their state, identity, or potential, and we overestimate the value of their actions because of that illusion. Our own ego works the same way. It blinds us, distorts our evaluations, and encourages us to reward ourselves or others for the wrong reasons.

Some even go the other way and present themselves in a lower state, using displays of desperation or victimhood to justify their actions, escape accountability, or shift blame onto others.

By stripping away ego and looking only at:

* the current state someone is in,
* the action they choose,
* and the consequences that follow,

we get a clearer way to interpret behaviour. We become better at understanding actions, predicting patterns, interacting with different states, and representing our own state in a way that earns the right evaluation, even if the actions are the same as before.

Seen this way, life becomes something we can optimize. Not in a cold mechanical sense, but in a practical one. We learn how to gather information cheaply, how to avoid being misled, how to invest wisely in others, and how to align our own actions with long-term returns.
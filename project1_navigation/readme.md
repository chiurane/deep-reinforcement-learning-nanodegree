#Project 1: Navigation
<hr>

**1. Overview**

We are going to train an agent to navigate and collect yellow bananas in a large square world which has both yellow
and blue bananas.

We have a typical reinforcement learning setup with states, actions, rewards between the environment an agent.

![picture alt](reinforcement_learning_problem.png)
The agent-environment interaction in reinforcement learning. (Source: Sutton and Barton, 2017)

###Rewards
A reward of +1 is awarded for collecting a yellow banana and reward of -1
is awarded for collecting a blue banana. Thus the goal of our agent is to collect
as many yellow bananas as possible while avoiding blue bananas.

###States
The state space has 37 dimensions and contains the agents velocity, along with
ray-based perception of objects around agent's forward direction. The agent uses this 
information to learn how to best select actions.

###Actions
The agent can select from four discrete actions corresponding to:
* 0 - move forward
* 1 - move backward
* 2 - turn left
* 3 - turn right

The task is episodic and in order to solve the environment, the agent must get an average score of +13 over 100 
consecutive episodes.

**2. Getting Started**

1. Download the environment from one of the links below. You need only select the environment that matches your operating system:

* Linux: [Click Here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip/ "Linux")

* Mac OSX: [Click Here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip/ "Mac OS")

* Windows (32-bit): [Click Here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip/ "Windows 32 bit")

* Windows (64-bit): [Click Here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip/ "Windows 64 bit")

(For Windows users) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

(For AWS) If you'd like to train the agent on AWS (and have not enabled a virtual screen), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the p1_navigation/ folder, and unzip (or decompress) the file.

**3. Project Instructions**

Follow the instructions in Navigation.ipynb to get started training the agents with the different provided algorithms.

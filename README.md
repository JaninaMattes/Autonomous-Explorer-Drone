<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

<!---
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
-->


<div align="center">
  <a href="https://github.com/JaninaMattes/Autonomous-Explorer-Drone/issues">
    <img src="https://img.shields.io/github/issues/JaninaMattes/Autonomous-Explorer-Drone.svg?style=for-the-badge" alt="Issues">
  </a>
  <a href="https://github.com/JaninaMattes/Autonomous-Explorer-Drone/blob/master/LICENSE">
    <img src="https://img.shields.io/github/license/JaninaMattes/Autonomous-Explorer-Drone.svg?style=for-the-badge" alt="License">
  </a>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  </a>
</div>



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/JaninaMattes/Autonomous-Explorer-Drone/blob/master/">
    <img src="img/drone/quadcopter.png" alt="Logo" width="350" height="220">
  </a>

  <h3 align="center">Autonomous Explorer Drone</h3>

  <p align="center">
    Exploring a learning-based method to autonomous flight.
    <br />
    <a href="https://github.com/JaninaMattes/Autonomous-Explorer-Drone/blob/master/Get_Started.md"><strong>Getting started »</strong></a>
    <br />
    <br />
    <!-- 
    <a href="https://github.com/othneildrew/Best-README-Template">View Demo</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Report Bug</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Request Feature</a>
    -->
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project

The design of a control system for an agile mobile robot in the continuous domain is a central question in robotics. This project specifically addresses the challenge of autonomous drone flight. Model-free reinforcement learning (RL) is utilized as it can directly optimize a task-level objective and leverage domain randomization to handle model uncertainty, enabling the discovery of more robust control responses. The task analyzed in the following is a single agent stabilization task.

### Drone Model and Simulation
The ```gym-pybullet-drones``` environment is based on the ```Crazyflie 2.x``` nanoquadcopter. It implements the
 ```OpenAI gym``` API for single or multi-agent reinforcement learning (MARL).

 <div align="center">
  <a href="https://github.com/JaninaMattes/Autonomous-Explorer-Drone/">
    <img src="img/drone/drone_config.png" alt="Logo" width="400" height="320">
  </a>
  <br>
<small>Fig. 1: The three types of <code>gym-pybullet-drones</code> models, as well as the forces and torques acting on each vehicle.</small>
</div>

### Training Result

The following shows a training result where the agent has learned to control the four independent rotors to overcome simulated physical forces (e.g. gravity) by the Bullet physics engine, stabilize and go into steady flight.

<div align="center">
  <a href="https://github.com/JaninaMattes/Autonomous-Explorer-Drone/">
    <img src="img/gifs/drone-flight-takeoff.gif" alt="Logo" width="400" height="320">
  </a>
  <br>
<small>Fig. 2: Rendering of a <code>gym-pybullet-drones</code> stable flight with a Crazyflie 2.x during inference.</small>
</div>

#### PPO Actor-Critic Architecture

In this project the policy gradient method is used for training with a custom implementation of [Proximal Policy Optimization (PPO)](https://arxiv.org/pdf/1707.06347.pdf).

<div align="center">
  <img src="img/architecture/architecture.png" alt="Logo" width="500">
  <br>
  <small>Fig. 3: Overview of the Actor-Critic Proximal Policy Optimisation Algorithm process</small>
</div>
</br>

The architecture consists of two separate neural networks: the actor network and the critic network. The actor network is responsible for selecting actions given the current state of the environment, while the critic network is responsible for evaluating the value of the current state.

The actor network takes the current state $s_t$ as input and outputs a probability distribution over the possible actions $a_t$. The network is trained using the actor loss function, which encourages the network to select actions that have a high advantage while also penalizing actions that deviate too much from the old policy. The loss function is defined as follows:

$$
L^{actor}(\theta) = \mathbb{E}_{t} \left[ \min\left(r_t(\theta) \hat{A}_t, \text{clip}\left(r_t(\theta), 1-\epsilon, 1+\epsilon\right) \hat{A}_t \right) \right]
$$

where $r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ is the probability ratio of the new and old policies, $\hat{A}_t$ is the estimated advantage function, and $\epsilon$ is a hyperparameter that controls how much the new policy can deviate from the old policy.

The critic network takes the current state $s_t$ as input and outputs an estimate of the value of the state $V_{\theta}(s_t)$. The network is trained using the critic loss function, which encourages the network to accurately estimate the value of the current state, given the observed rewards and the estimated values of future states. The loss function is defined as follows:

$$
L^{critic}(\theta) = \mathbb{E}_{t} \left[ \left(V_{\theta}(s_t) - R_t\right)^2 \right]
$$

where $R_t$ is the target value for the current state, given by the sum of the observed rewards and the estimated values of future states.

#### Action and Observation Space

The observation space is defined through the quadrotor state, which includes the position, linear velocity, angular velocity, and orientation of the drone. The action space is defined by the desired thrust in the z direction and the desired torque in the x, y, and z directions.

#### Reward Function

The reward function defines the problem specification as follows:

$$
\text{Reward} =
\begin{cases}
-5, & \text{height} < 0.02 \\
-\frac{1}{10 \cdot y_{pos}}, & \text{height} \geq 0.02
\end{cases}
$$

where $y_{pos}$ is the current height of the drone. The reward function encourages the drone to maintain a certain height while also penalizing excessive movement in the y-axis.


<p align="right">(<a href="#readme-top">back to top</a>)</p>

### PyBullet Environment & Drone

#### Environment

The environment is a custom OpenAI Gym environment built using PyBullet for multi-agent reinforcement learning with quadrotors.

<div align="center">
  <a href="https://github.com/JaninaMattes/Autonomous-Explorer-Drone/">
    <img src="img/pybullet.png" alt="Logo" width="450" height="">
  </a>
  </br>
  <small>Fig. 4: 3D simulation of the drone's orientation in the x, y, and z axes.</small>
</div>

#### PID Controller

To stabilize the quadrotor during simulation, a **pre-implemented classical PID flight controller** is used.  
The controller follows a **cascaded control architecture**, which is standard for real-world micro aerial vehicles such as the **Crazyflie 2.x**.

In the `gym-pybullet-drones` environment, the PID controller operates directly on the simulated vehicle state provided by **PyBullet** and generates low-level motor commands to ensure stable and physically consistent flight.

---

##### Cascaded Control Structure

The controller is organized as a **hierarchical (cascaded) PID system**:

1. **Position Control (Outer Loop)**  
   Regulates the drone’s Cartesian position and computes desired roll, pitch, and collective thrust setpoints.

2. **Attitude Control (Inner Loop)**  
   Regulates roll, pitch, and yaw angles and outputs desired body torques.

3. **Motor Mixing**  
   Converts thrust and torque commands into individual motor speeds for the four rotors.

This architecture closely mirrors the **onboard Crazyflie PID controller** used in real flight.

---

##### PID Control Law

Each control loop applies the standard PID formulation:

\[
u(t) = K_p \, e(t) + K_i \int_0^t e(\tau)\, d\tau + K_d \frac{d}{dt} e(t)
\]

where:

- \( e(t) \) is the error between the desired setpoint and the measured state  
- \( K_p \) is the proportional gain  
- \( K_i \) is the integral gain  
- \( K_d \) is the derivative gain  

The controller continuously computes these errors using simulated onboard sensor data, including position, velocity, orientation, and angular rates.

---

##### Controlled Quantities

The PID controller stabilizes and regulates:

- **Position:** \( x, y, z \)
- **Attitude:** roll, pitch, yaw
- **Angular rates**
- **Collective thrust**

The resulting control signals are translated into **four individual rotor thrust commands**, accounting for the Crazyflie 2.x quadrotor configuration and motor layout.

---

##### Role in the Learning Pipeline

The PID controller serves multiple purposes:

- Baseline stabilizing controller
- Reference policy for comparison with reinforcement learning agents
- Reliable mechanism for hovering, takeoff, and trajectory tracking

This separation allows reinforcement learning methods (e.g., PPO) to focus on **high-level decision making**, while low-level stabilization remains robust and physically grounded.

---

<div align="center">
  <img src="img/gifs/pid-controller-mechanism.gif" alt="PID controller mechanism" width="500" height="320">
  <br>
  <small>
    Fig. 2: Stable Crazyflie 2.x flight achieved using a cascaded PID controller in the
    <code>gym-pybullet-drones</code> simulation.
  </small>
</div>


<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Built With

The project was developed using Python and the PyTorch machine learning framework. To simulate the quadrotor's environment, the Bullet physics engine is leveraged. Further, to streamline the development process and avoid potential issues, the pre-built PyBullet drone implementation provided by the [gym-pybullet-drones library](https://github.com/utiasDSL/gym-pybullet-drones) is utilized.


Programming Languages-Frameworks-Tools<br /><br />
[![My Skills](https://skillicons.dev/icons?i=py,pytorch,docker,anaconda,unity&theme=light&perline=10)](https://skillicons.dev)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may be setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Requirements and Installation

This repository was written using Python 3.10 and Anaconda tested on macOS 14.4.1.

#### Project structure

The repository is organized so that core algorithms, simulation backends, and optional extensions are separated:

```
Autonomous-Explorer-Drone/
├── src/                       # Main project source code (PPO, training, utils)
├── gym_pybullet_drones/       # PyBullet-based drone simulation (vendored)
├── unity_mlagent_drones/      # Optional Unity ML-Agents extension (future work)
├── assets/                    # Models, images, and videos
├── docs/                      # PDFs and theoretical references
├── img/                       # Figures used in README/docs
├── requirements*.txt          # Python dependencies
├── Dockerfile                 # Reproducible environment
└── README.md                  # Project description
```
Design rationale:

* All original research and training code lives in src/

* Simulation environments are kept isolated to avoid coupling and ease replacement

* Unity support is optional and does not affect the core PyBullet Gym pipeline


#### Installation

_Major dependencies are gym, pybullet, stable-baselines3, and rllib_

1. Create virtual environment and install all major dependencies 
   ```
    $ pip3 install --upgrade numpy matplotlib Pillow cycler 
    $ pip3 install --upgrade gym pybullet stable_baselines3 'ray[rllib]' 
   ```

   or requirements.txt
   ```
   $ pip install -r requirements_pybullet.txt
   ```

2. Video recording requires to have ```ffmpeg``` installed, on macOS
   ```
   $ brew install ffmpeg
   ```

   or on Ubuntu
   ```
   $ sudo apt install ffmpeg
   ```

3. The ```gym-pybullet-drones``` repo is structured as a Gym Environment and can be installed with pip install --editable

   ```
   $ cd gym-pybullet-drones/
   $ pip3 install -e .
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- ROADMAP -->
## Roadmap

- [x] Add Changelog
- [x] Add back to top links
- [ ] Fix sparse reward issue by adding prox rewards
- [ ] Adjustment of the reward function to achieve the approach of a target
- [ ] Implement in Unity with ML agents
- [ ] Adjust Readme file

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Project Link: [Autonomous-Explorer-Drone](https://github.com/JaninaMattes/Autonomous-Explorer-Drone/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [GitHub - ML Drone Collection](https://github.com/mbaske/ml-drone-collection)
* [Lab ML AI - PPO Explained](https://nn.labml.ai/rl/ppo/)
* [Huggingface - PPO Explained](https://huggingface.co/blog/deep-rl-ppo)
* [Pytorch Forum - Discussion](https://discuss.pytorch.org/t/understanding-log-prob-for-normal-distribution-in-pytorch/73809)
* [David Silver - Introduction to Reinforcement Learning](https://www.davidsilver.uk/teaching/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[take-off-aviariy-gif]: images/gifs/drone-flight-takeoff.gif
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 

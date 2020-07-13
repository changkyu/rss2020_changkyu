# Learning to Slide Unknown Objects with Differentiable Physics Simulations

Changkyu Song and Abdeslam Boularias, Learning to Slide Unknown Objects with Differentiable Physics Simulations, Robotics: Science and Systems (R:SS), July 14-16th, 2020, Oregon State University at Corvallis, Oregon, USA. [[PDF]](https://arxiv.org/pdf/2005.05456.pdf) [[Video]](https://www.youtube.com/watch?v=cHNnJNRQBPc) [[Project Page]](https://sites.google.com/site/changkyusong86/research/rss2020)

We propose a new technique for pushing an unknown object from an initial configuration to a goal configuration with stability constraints. The proposed method leverages recent progress in differentiable physics models to learn unknown mechanical properties of pushed objects, such as their distributions of mass and coefficients of friction. The proposed learning technique computes the gradient of the distance between predicted poses of objects and their actual observed poses, and utilizes that gradient to search for values of the mechanical properties that reduce the reality gap. The proposed approach is also utilized to optimize a policy to efficiently push an object toward the desired goal configuration. Experiments with real objects using a real robot to gather data show that the proposed approach can identify mechanical properties of heterogeneous objects from a small number of pushing actions. 

# Results

<img src=".readme/img/rss2020_qual.png" width="100%"/>

<img src=".readme/img/real_book_sims_vs_cell.png" width="33%"/><img src=".readme/img/real_hammer_sims_vs_cell.png" width="33%"/>
<img src=".readme/img/real_box_sims_vs_cell.png" width="33%"/><img src=".readme/img/real_snack_sims_vs_cell.png" width="33%"/><img src=".readme/img/real_windex_sims_vs_cell.png" width="33%"/>
<img src=".readme/img/plan_grasp.png" width="33%"/><img src=".readme/img/plan_nsims.png" width="33%"/>

# Author

[Changkyu](https://sites.google.com/site/changkyusong86) (changkyusong86@gmail.com)



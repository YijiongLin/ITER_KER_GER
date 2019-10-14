# ITER_KER_GER
## Description
This repo refers to the paper [*Invariant Transform Experience Replay*](https://arxiv.org/abs/1909.10707#), which had been submitted to ICRA-2020. 

Deep reinforcement learning (DRL) is a promising approach for adaptive robot control, but its current application to robotics is currently hindered by high sample requirements. We propose two novel data augmentation techniques for DRL based on invariant transformations of trajectories in order to reuse more efficiently observed interaction. The first one called Kaleidoscope Experience Replay exploits reflectional symmetries, while the second called Goal-augmented Experi- ence Replay takes advantage of lax goal definitions. In the Fetch tasks from OpenAI Gym, our experimental results show a large increase in learning speed

And this repo is built on top of [OpenAI Baselines](https://github.com/openai/baselines/tree/master/baselines) and [OpenAI Gym](https://github.com/openai/gym). 

This implementation requires the installation of the [OpenAI Baselines](https://github.com/openai/baselines/tree/master/baselines) module. After the installation, please download all the files in folder "her" of this repo and then copy them to override the one in baselines/her.

## Usage
To reproduce the results in our paper, please run :
```
python -m baselines.run --alg=her --env=FetchPickAndPlace-v1 --num_timesteps=1e6 --n_cycles=100 --save_path=/home/user/policies/her/iter --log_path=/home/bourne/log_data/her/iter --before_PER_minibatch_size=256 --n_rsym=8 --n_PER=4
```

options include:
* `--num_cpu`: Number of cpus. The paper uses 19 cpus (as in the [original paper](https://arxiv.org/abs/1802.09464) presenting this HER implementation. Please note that as the HER's author said, running the code with different cpus is NOT equivalent, for more info please check (here)[https://github.com/openai/baselines/issues/314].
* `--env`: string of the gym_flowers env. Possible choices are *FetchPickAndPlace-v1, FetchSlide-v1, FetchPush-v1*. (There will be more choices on Baxter robot in the near future, please keep watching on our repo :). )
* `--before_PER_minibatch_size`: To specify the original minibatch size.
* `--n_rsym`: To specify how many reflectional planes you would like to augment the samples. The additional number of symmetrical samples is $2*n_rsym-1$.
* `--n_PER`: To specify the hyperparameter of GER.


## More Information
For more information please check:
1. [Website](http://www.juanrojas.net/ker/)
2. [Paper](https://arxiv.org/abs/1909.10707#)
3. [Youtube](https://www.youtube.com/watch?v=qM3QEeqHTdk&feature=youtu.be), [Youku](https://v.youku.com/v_show/id_XNDM3NDY0NzM0MA==.html?spm=a2hzp.8244740.0.0)

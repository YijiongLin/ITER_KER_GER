# ITER_KER_GER

## Description

This repo refers to the paper [*Invariant Transform Experience Replay*](https://arxiv.org/abs/1909.10707#), which had been submitted to ICRA-2020. 

Deep reinforcement learning (DRL) is a promising approach for adaptive robot control, but its current application to robotics is currently hindered by high sample requirements. We propose two novel data augmentation techniques for DRL based on invariant transformations of trajectories in order to reuse more efficiently observed interaction. The first one called Kaleidoscope Experience Replay exploits reflectional symmetries, while the second called Goal-augmented Experi- ence Replay takes advantage of lax goal definitions. In the Fetch tasks from OpenAI Gym, our experimental results show a large increase in learning speed

And this repo is built on top of [OpenAI Baselines](https://github.com/openai/baselines/tree/master/baselines) and [OpenAI Gym](https://github.com/openai/gym). 

## Installation

This implementation requires the installation of the [OpenAI Baselines](https://github.com/openai/baselines/tree/master/baselines) module. 
After the installation, please create a new folder for this repo and go inside.
```
mkdir ITER_KER_GER && cd $_
```
Download all the codes held in this repo.
```
git clone git@github.com:birlrobotics/ITER_KER_GER.git
```
Finally, please copy the files held in folder `ITER_KER_GER/her` and paste into `baselines/baselines/`.
```
copy -rf her ~/baselines/baselines/
```
## Usage

To reproduce the results in our paper, please run :
```
python -m baselines.run --alg=her --env=FetchPickAndPlace-v1 --num_timesteps=1e6 --n_cycles=100 --save_path=/home/user/policies/her/iter --log_path=/home/user/log_data/her/iter --before_PER_minibatch_size=256 --n_rsym=8 --n_PER=4
```

options include:
* `--num_cpu`: Number of workers(threads/cpus). The results in our paper just used 1 worker in order to show the significant improvements in learning speed. The [original HER paper](https://arxiv.org/abs/1802.09464) presents this HER implementation. (**Please note that as the HER's author said, running the code with different cpus is NOT equivalent. For more information about this issue, please check [here](https://github.com/openai/baselines/issues/314).**)
* `--env`: To specify the experimental environment in each run. Possible choices are *FetchPickAndPlace-v1, FetchSlide-v1, FetchPush-v1*. (There will be more choices on Baxter robot in the near future, please keep watching on our repo :). )
* `--before_PER_minibatch_size`: To specify the original minibatch size.
* `--n_rsym`: To specify the hyperparameter of KER. More specifically, it is to specify how many reflectional planes you would like to augment the samples. For more information, please checkout our [Paper](https://arxiv.org/abs/1909.10707#).
* `--n_PER`: To specify the hyperparameter of GER. More specifically, it is to specify how many transitions' goals you would like to augment. For more information, please checkout our [Paper](https://arxiv.org/abs/1909.10707#).
* `--log_path`: To specify the log file saved path.
* `--save_path`: To specify the policy parameters saved path.

## Loading and visualizing models

This [page](https://github.com/openai/baselines#saving-loading-and-visualizing-models) from OpenAI Baselines has a good indicaition on loading and visualizing models.

## More Information

For more information please check:
1. [Website](http://www.juanrojas.net/ker/)
2. [Paper](https://arxiv.org/abs/1909.10707#)
3. [Youtube](https://www.youtube.com/watch?v=qM3QEeqHTdk&feature=youtu.be), [Youku](https://v.youku.com/v_show/id_XNDM3NDY0NzM0MA==.html?spm=a2hzp.8244740.0.0)

## Credits

`ITER_KER_GER` is maintained by the BIRL Intelligent Manipulation team. Contributors include:

- Yijiong Lin (Bourne), yijiong.lin@bristol.ac.uk
- Jiancong Huang (Jim), 374729746@qq.com (currently looking for Ph.D. position)

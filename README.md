# nnsrm-neurips18

This is the repo for ... 
```
Lu, Q., Chen, P.-H., Pillow, J. W., Ramadge, P. J., Norman, K. A., & Hasson, U. (2018). 
Shared Representational Geometry Across Neural Networks. arXiv [cs.LG]. 
Retrieved from http://arxiv.org/abs/1811.11684
```

... and the shared response model (SRM) is implemented in <a href="https://github.com/brainiak/brainiak">BrainIAK</a>. 

### What is this project about?

- 1 sentence summary: **different neural networks with the same learning experience acquire representations of the same "shape"** 

- Here's a <a href="https://qihongl.github.io/nnsrm-NeurIPS18.html">5 mins version of that paper</a>. 

- Here's a <a href="https://github.com/qihongl/demo-nnalign">tutorial</a> that describe the minimal analysis pipeline. I'm working on converting this to a jupyter binder or something interactive...


### Doc: 

#### Files for the simulation (under `simulation/`): 

- `run_sim.ipynb`: run the simulation described in the paper
- `data_gen.py`: make toy data set to train NNs
- `models.py`: define a simple neural network


#### Files for the experiments (Files under the root dir): 

*The notebooks are not runnable yet, since they depend on some pre-computed data. I'm working on an easy way of hosting the data publicly. Though re-running the whole analysis should be possible. 

- `show_*.ipynb`: load some pre-computed data (e.g. activity from some pre-trained neural networks), apply certain analyses (e.g. SRM), then plot the results 
- `train_*.py`: train some models (e.g. conv nets) on some dataset (e.g. cifar10)
- `save_acts_cifar.py`: test and save neural network activity matrices 
- `run_analyses.py`: run SRM, RSA, etc. 
- `models.py`: some models (e.g. conv nets)
- `resnet.py`: resnets from <a href="https://github.com/raghakot/keras-resnet">raghakot/keras-resnet</a> [2]
- `config.py`: define some constants, such as how to re-arrange the ordering of the images in cifar
- `data_loader.py`: util for loading data 


#### Other files: 
- `qmvpa`: contains some analyses util functions

### Dependencies/References: 

[1] <a href="https://github.com/philipperemy/keract">philipperemy/keract</a>  
[2] <a href="https://github.com/raghakot/keras-resnet">raghakot/keras-resnet</a>  
[3] <a href="https://github.com/ContextLab/hypertools">hypertools</a>  
[4] <a href="https://github.com/brainiak/brainiak">BrainIAK</a>  

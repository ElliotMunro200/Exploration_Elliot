This is a PyTorch implementation of the Exploration paper 
"Learning to Explore by Reinforcement over High-Level Options" by Juncheng Liu et al. 2021
The paper describes a hirachical RL method of using both a look-around and a frontier-navigation option

# The code is based on the Active Neural SLAM implementation
[Learning To Explore Using Active Neural SLAM](https://openreview.net/pdf?id=HklXn1BKDH)<br />
Devendra Singh Chaplot, Dhiraj Gandhi, Saurabh Gupta, Abhinav Gupta, Ruslan Salakhutdinov<br />
Carnegie Mellon University, Facebook AI Research, UIUC

Project Website: https://devendrachaplot.github.io/projects/Neural-SLAM


## Installing Dependencies
We use earlier versions of [habitat-sim](https://github.com/facebookresearch/habitat-sim) and [habitat-api](https://github.com/facebookresearch/habitat-api). The specific commits are mentioned below.

Installing habitat-sim:
```
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim; git checkout 9575dcd45fe6f55d2a44043833af08972a7895a9; 
pip install -r requirements.txt; 
python setup.py install --headless
python setup.py install # (for Mac OS)

```

Installing habitat-api:
```
git clone https://github.com/facebookresearch/habitat-api.git
cd habitat-api; git checkout b5f2b00a25627ecb52b43b13ea96b05998d9a121; 
pip install -e .
```

Install pytorch from https://pytorch.org/ according to your system configuration. The code is tested on pytorch v1.2.0. If you are using conda:
```
conda install pytorch==1.2.0 torchvision cudatoolkit=10.0 -c pytorch #(Linux with GPU)
conda install pytorch==1.2.0 torchvision==0.4.0 -c pytorch #(Mac OS)
```

## Setup
Clone the repository and install other requirements:
```
cd Exploration;
pip install -r requirements.txt
```

The code requires datasets in a `data` folder in the following format (same as habitat-api):
```
Neural-SLAM/
  data/
    scene_datasets/
      gibson/
        Adrian.glb
        Adrian.navmesh
        ...
    datasets/
      pointnav/
        gibson/
          v1/
            train/
            val/
            ...
```
Please download the data using the instructions here: https://github.com/facebookresearch/habitat-api#data

To verify that dependencies are correctly installed and data is setup correctly, run:
```
python main.py -n1 --auto_gpu_config 0 --split val
```


## Usage

### Training:
For training the model on the Exploration task: (see arguments.py for more details)
```
python main.py
```

### For evaluation:
For evaluating the pre-trained models:
```
python main.py --split val --eval 1 --load_global tmp/models/exp1/model_best.global
```

For visualizing the agent observations and predicted map and pose, add `-v 1` as an argument to the above
# IEDQN: Information Exchange DQN with a Centralized Coordinator for Traffic Signal Control

This repo implements the [IEDQN](https://ieeexplore.ieee.org/document/9206820 ) algorithm for traffic signal control in SUMO-simulated environments. We supply a 5$\times$5 traffic grid as the  environment.

## Requirement

* Python == 3.7.0
* TensorFlow == 1.14.0
* SUMO >= 0.32.0

## Usages

1. To train the agent, run

   ````shell
   python main.py --base-dir [base-dir] --port [port] train --config-dir [config-dir]
   ````

2. To evaluate and compared trained agents, run

   ````shell
   python main.py --base-dir [base-dir] evaluate --agents [agents] --evaluation-seeds [seeds]
   ````

## Citation

If you find this repo is useful in your research, please cite our paper "[IEDQN: Information Exchange DQN with  a Centralized Coordinator for Traffic Signal Control](https://ieeexplore.ieee.org/document/9206820 )".

````
@INPROCEEDINGS{9206820,
  author={Xie, Donghan and Wang, Zhi and Chen, Chunlin and Dong, Daoyi},
  booktitle={2020 International Joint Conference on Neural Networks (IJCNN)}, 
  title={IEDQN: Information Exchange DQN with a Centralized Coordinator for Traffic Signal Control}, 
  year={2020},
  volume={},
  number={},
  pages={1-8},
  doi={10.1109/IJCNN48605.2020.9206820}}
````

## Aknowledgement

This repo is based on the environment framework implemented in repo [MA2C](https://github.com/cts198859/deeprl_signal_control).

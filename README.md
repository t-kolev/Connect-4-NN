# Connect 4 Software agent

This repository contains the implementation of a software agent designed to play the game Connect 4. It was developed as a task for the "Programming Project in Python" course at TU Berlin. The primary objective was to implement and design an intelligent agent capable of playing Connect 4 using the reinforcement learning.

## System Requirements

### Interpreter
* Python 3.10 (Anaconda)

### Libraries
For all the following libraries the newest version was installed.
* numpy
* time
* typing
* pickle
* scikit-learn
* tensorflow
* random
* enum
* os

## Usage

### Already trained data
As we (finally) managed to share the training data from other computers, our trained neural network can already be 
used. In order to do so the instructions from "Evaluation" can be applied straight away.

### Training
IMPORTANT: If it is desired to train the neural network from the beginning, first the function 
``nn.NeuralNetwork().reset()`` has to be run in main. This however will cause all the previously trained data to 
disappear.

First the function ``train_evaluate(n)`` should be run (in main), where ``n`` represents the number of training runs.
Each run consists of 5 trainings of the Neural Network against the UCT agent. After that the performance is
measured against the random agent (10 games), minimax agent (2 runs) and the UCT agent (10 runs).

If however only the training is desired then only the function ``train_nn_vs(ucta.generate_move, r)`` should be
executed in main. ``r`` represents the number of runs against the chosen agent, we recommend to train against the
UCT agent, however a different agent can be chosen.

### Evaluation
Afterwards either the ``evaluate_nn()`` function can be chosen to measure the performance of the agent or the user
can choose to play against the trained Neural Network by running
``human_vs_agent(ucta.generate_move, args_1=(100, True))`` in the main. This function can also be executed in the
following form: ``human_vs_agent(ucta.generate_move, args_1=(100, True), x)``, where ``x`` represents the agent
that the neural network should play against.

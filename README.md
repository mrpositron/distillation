# Self-Distillation and Knowledge Distillation Experiments

Detailed report can be found [here](https://wandb.ai/mrpositron/cifar10_sd/reports/Self-Distillation-and-Knowledge-Distillation-Experiments--VmlldzoxNTEwOTQ0).

## Introduction

There are three mysteries in deep learning. Highlighted in the recent blog by Zeyuan Allen-Zhu and Yuanzhi Li [1]. They are: ensemble, self-distillation, and knowledge distillation. In the blog they discuss the mysteries around them and analyze why they actually work. Inspired by the fact that these ideas work, and motivated to validate them. I ran some experiments on my own. Below you can see the results. 

![Taken from the Microsoft Post [1]](./mf.gif)

However, let me briefly describe what those techniques mean. First technique is **ensemble**. We train multiple classifiers independently and then average all results. Second technique is **knowledge distillation**. We take the output of the ensemble and use it as a target to another model. Here the target is a probability distribution. Final technique is **self-distillation**. We train one classifier, and use its probability distribution from the softmax to another classifier.

## Training

Due to the hardware constraints the architecture to perform experiments was chosen to be ResNet-18. The network was trained 5 times with different random seeds for initialization. Hyperparameters details can be seen below:

| Hyperparameters  | Values        |
|------------------|---------------|
| Loss             | Cross Entropy |
| Learning Rate    | 0.001         |
| Optimizer        | Adam          |
| Number of epochs | 100           |
| Training size    | 45000         |
| Validation size  | 5000          |


## Ensemble

In the table below you can see the accurcacies of each individual model on the validation and test datasets.

| seed | validation accuracy | test accuracy |
|------|---------------------|---------------|
| 0    |               86.66 |         86.01 |
| 1    |               86.06 |         85.46 |
| 2    |               86.08 |         85.71 |
| 3    |               86.38 |         86.17 |
| 4    |                  87 |         86.39 |

When we combine these models and average the result. The accuracy of the ensemble model on the test set is 89.07.

## Knowledge Distillation

The validation accuracy obtained with knowledge distiallation is 87.39, and the test accuracy is 87.

## Self-Distillation

In the table below you can see the accuracies produced by the self-distillaion.

| seed | validation accuracy | test accuracy | seed | validation accuracy | test accuracy |
|------|---------------------|---------------|------|---------------------|---------------|
| 0    |               86.66 |         86.01 | 10   |                87.7 |         86.88 |
| 1    |               86.06 |         85.46 | 11   |               87.54 |         86.48 |
| 2    |               86.08 |         85.71 | 12   |               87.14 |         86.13 |
| 3    |               86.38 |         86.17 | 13   |               86.46 |         85.32 |
| 4    |                  87 |         86.39 | 14   |               87.38 |         87.08 |

The table shows the results of using the model on the left as a teacher (TN, i.e. Teacher Network), and student (SN, i.e. Student Network)on the right. Models on the left are identical to those used before.

## Discussion

It is clear to see that ensemble, knowledge and self-distillation works. The accuracy on the test set goes up for about ~1% using self-distillation. The ensemble model gives us ~2.5-3% boost.
Knowledge distillation works as seen from the results. However, I expected more significant boost. One interesting thing to note is that if we will create an ensemble from the models produced in self-distillation, then the accuracy will be 89.38.



## References

[1] - Three mysteries in deep learning: Ensemble, knowledge distillation, and self-distillation. By Zeyuan Allen-Zhu , Senior Researcher  Yuanzhi Li , Assistant Professor, Carnegie Mellon University. ([link](https://www.microsoft.com/en-us/research/blog/three-mysteries-in-deep-learning-ensemble-knowledge-distillation-and-self-distillation/))


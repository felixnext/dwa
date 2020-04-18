# Pay attention to the Task: Avoid forgetting in sequential learning through dynamic weight allocation

> Note: Repository is forked from https://github.com/joansj/hat
>
> Reference for original code: Serrà, J., Surís, D., Miron, M. & Karatzoglou, A.. (2018). Overcoming Catastrophic Forgetting with Hard Attention to the Task. Proceedings of the 35th International Conference on Machine Learning, in PMLR 80:4548-4557

## Abstract

While learning multiple tasks, especially if trained in sequential order, classic learning techniques for DNNs tend to leverage the
weights of the network in a suboptimal manner. This results in the overwriting of relevant information and in turn to catastrophic forgetting.
Such information loss limits the capabilities for real-world learning of
these network architectures and remains one of the hurdles towards artifical general intelligence. In this paper, we introduce a novel attention
based technique that learns a semantic distribution of information across
network weights. Our approach generates weight masks during inference,
allowing the network to choose optimal weight configurations based on
input type classification. We show that our approach outperforms existing
techniques (such as EWC and HAT) on sequentially learning multiple
datasets and that it is robust against multiple common network topologies. Furthermore, we show that it learns a semantic understanding of
weight relevance w.r.t. task information, which we believe can also aid
generalization and one-shot learning.

## Authors

Felix Geilert

## Reference and Link to Paper

TODO


## Installing

1. Create a python 3 conda environment (check the requirements.txt file)

2. The following folder structure is expected at runtime. From the git folder:
    * src/ : Where all the scripts lie (already produced by the repo)
    * dat/ : Place to put/download all data sets
    * res/ : Place to save results
    * tmp/ : Place to store temporary files

3. The main script is src/run.py. To run multiple experiments we use src/run_multi.py or src/work.py; to run the compression experiment we use src/run_compression.sh.

## Notes

* If using this code, parts of it, or developments from it, please cite the above reference. 
* We do not provide any support or assistance for the supplied code nor we offer any other compilation/variant of it. 
* We assume no responsibility regarding the provided code.


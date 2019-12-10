# Asynchronous Parallel Gradient Boosting using Parameter Server

## URL
The link to our web page is: https://gavinl99.github.io/15418-Project/

## Summary
We plan on implementing a gradient boosting decision tree (GBDT) algorithm in an asynchronous framework. We are going to distribute the work in parallel through a parameter server, whilst first creating a proof of concept in OpenMP and MPI. Performance will be compared with sequential and parallel implementation in OpenCV and XGBoost.

## Background
Gradient boosting is a machine learning algorithm that ensembles weak learners like decision stumps and improves accuracy. The basic idea is to incrementally train a model to minimize the empirical loss function over the function space by fitting a weak learner that points in the negative gradient direction. We will also use decision trees in our implementation of gradient boosting.

Many researchers have proposed various ways to parallelize GBDT algorithm by generating good subsample of the original dataset and have worker nodes train weak learners, usually decision trees, on each subset. Explicit synchronization will be done at the end of every iteration to aggregate all trees built. However, this fork-join paradigm fails to scale as a small number of slow worker nodes can significantly slow down the training. For example, LightGBM, the state-of-art parallel GBDT framework, usually only achieve 5x to 7x speedup on a 32-core machines.

That's where the asynchronous parallel GBDT with parameter server framework comes to rescue: the server receives trees from workers, and workers build trees on subsamples of dataset asynchronously, which allows overlapping of comminucation and computation time.

## Challenge
There are two major challenges of implementing async-GBDT algorithm:
* How to build the parameter server for workers to pull and commit their work.
* How to ensure load balancing such that the server node itself will not become the bottleneck.

Our proposed solution is to have multiple server nodes instead of one and implement a shared work queue under the producer-consumer framework. Having multiple server node incurs more communication costs, so we need to experiment and do profiling to find the best configuration of server / worker nodes.

Additionally we need to tune every gradient boosting algorithm on the ratio of data to train and to test on. As a result, some algorithms may lose accuracy in order to remain faster or vice versa.

## Resources
There are various synchronous implementations of gradient boosting algorithm using fork-join parallel method. We refer to the following papers for efficient implementation of stochastic gradient boosting and delayed gradient descent (DC-ASGD): 

\text{[1]} J. H. Friedman, “Stochastic gradient boosting,” Computational Statistics Data Analysis, vol. 38, no. 4, pp. 367–378, 2002.

\text{[2]} J. Ye, J. H. Chow, J. Chen, and Z. Zheng, “Stochastic gradient boosted distributed decision trees,” in Acm Conference on Information Knowl- edge Management, 2009, pp. 2061–2064.

We are going to draw inspiration for the base algorithm from the following article by Rory Mitchell describing a gradient boosting implementation through CUDA:

\text{[3]} \href{https://devblogs.nvidia.com/gradient-boosting-decision-trees-xgboost-cuda/}{Gradient Boosting, Decision Trees and XGBoost with CUDA, Nvidia}

From that article, we will also use the XGBoost implementation outlined to compare our results with the current standard implementation.

We mainly refer to the following paper for asynchronous implementation of GBDT and We plan to follow the parameter server framework outlined there. 

\text{[4]} \href{https://www.cs.cmu.edu/~muli/file/parameter_server_osdi14.pdf}{Daning, Fen, et al. "Asynch-SGBDT: Train a Stochastic Gradient Boosting Decision Tree in an Asynchronous Parallel Manner"}

## Goals and Deliverables
Through the process of completing the project, we plan on having finished the following:
* Have completed the OpenMP implementation
* Have completed the MPI implementation
* Have implemented the asynchronous GBDT with parameter server
* Experiment on different configurations of parameter server
* Compare the performance of async-GBDT with sequential / parallel implementations of our own and standard libraries like OpenCV and XGBoost on different types of dataset (size, sparsity, dimensions, etc.)
* Compare each implementation of GBDT in terms of speedup, communication time and accuracy

If all base goals are achieved, we plan on implementing additional parallel implementations using both an asynchronous and synchronous framework on CUDA. We believe that by doing so, we will have a more complete understanding of each parallel programming model's advantages and disadvantages.

Additionally, we hope to use more than one data set for training and testing if we have enough time, in order to determine if a specific implementation favours a specific type of data set or machine learning task, such as regression versus ranking or classification.

For the poster session, we plan of showing the specific characteristics of each implementation in terms of speedup, accuracy and overheads. As such, we hope to create a comprehensive table of which implementation is favoured under a specific set of criteria and data.

Specifically, we hope to show that creating an asynchronous implementation significantly reduces communication overhead and speeds up the learning process significantly as data size increases.

## Platform Choice
As mentioned before, we are planning to use both OpenMP and MPI in C++ to implement the initial synchronous version of gradient boosting. Both those implementations work to outline the main aspects that may cause a synchronous implementation of gradient boosting to lack the required speedup.

We will also use C++ for the asynchronous implementation, as it is easy to implement and fast enough to parse the data and facilitate the communication we require.

## Schedule
We plan to achieve the following by the given dates:

| Date        | Goal Reached           |
| ------------- |:-------------:|
| 11/3     | Research on fork-join GBDT, async-GBDR and find datasets to train and test on |
| 11/10     | Implement the OpenMP gradient boosting algorithm |
| 11/15     | Implement the MPI gradient boosting algorithm |
| 11/20     | Implement the asynchronous parameter server and associated data structures |
| 11/27     | Implement and integrate the asynchronous GBDT algorithm |
| 11/30     | Experiments and profiling to optimize parameter server |
| 12/06    | Comparison analysis, write-up and poster |

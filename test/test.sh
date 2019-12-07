#!/bin/bash
../src/cpp/gbdt_train --feature_size 3 --train_file ../data/train.txt --max_depth 4 --shrinkage 0.1 --feature_ratio 1.0 --data_ratio 1.0 --debug true --min_leaf_size 0 --loss LogLoss --num_of_threads 16 --use_async true --tree_sample 0.3 --num_trees 500

#!/bin/bash
./gbdt_train --feature_size 150360 --train_file ../data/E2006.train --max_depth 3 --shrinkage 0.1 --feature_ratio .01 --data_ratio 1.0 --debug true --min_leaf_size 0 --num_of_threads 16 --use_async true --tree_sample 0.3 --num_trees 500


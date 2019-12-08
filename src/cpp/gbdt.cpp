// Author: qiyiping@gmail.com (Yiping Qi)
#include "gbdt.hpp"
#include "math_util.hpp"
#include "util.hpp"
#include "auc.hpp"
#include "loss.hpp"
#include <iostream>
#include <cmath>
#include <cassert>
#include <algorithm>
#include "time.hpp"
#include <random>


//#define NUM_INDEP_TREES 8
//#define TREE_SAMPLE_THRESHOLD 0.3
#define PRINT_TREE_INTERVAL 50

namespace gbdt {
  GBDT::~GBDT() {
    ReleaseTrees();
    trees_vec_.destroy_all();
    delete[] gain;
  }

  void GBDT::Init(DataVector &d) {
    // this computes the weighted mean as init guess
    bias = conf.loss->GetBias(d, d.size());
    trees = new RegressionTree *[conf.num_trees];
    for (int i = 0; i < conf.num_trees; ++i) {
      trees[i] = new RegressionTree(conf);
    }
  }

  // for out-of-sample prediction
  ValueType GBDT::Predict(const Tuple &t) const {
    if (!trees)
      return kUnknownValue;

    ValueType r = bias;
    if (conf.enable_initial_guess) {
      r = t.initial_guess;
    }

    for (size_t i = 0; i < conf.num_trees; ++i) {
      r += shrinkage * trees[i]->Predict(t);
    }

    return r;
  }

  ValueType GBDT::PredictAsync(const Tuple &t, RegressionTree *tree, ValueType temp_pred) const {
    temp_pred += shrinkage * tree->Predict(t);
    return temp_pred;
  }

  /**
   * Prediction from prev iteration
   * @param t
   * @param n: idx of iteration, only update this iteration
   * @param temp_pred: cumulatively updated result for each data point
   * @return
   */
  ValueType GBDT::Predict_OMP(const Tuple &t, size_t n, ValueType temp_pred) const {
    if (!trees)
      return kUnknownValue;

    if (n == 0) {
      if (conf.enable_initial_guess) {
        return t.initial_guess;
      }
      return bias;
    }
    size_t start = (n - 1) * conf.num_of_threads;
    size_t end = conf.num_trees < n * conf.num_of_threads? conf.num_trees: n * conf.num_of_threads;

    for (size_t i = start; i < end; ++i) {
      temp_pred += shrinkage * trees[i]->Predict(t);
    }

    return temp_pred;
  }

  /*
   * Generate a random sample of index first
   * Take read lock and fit a new tree
   * push it to the forest vector
   */
  void GBDT::WorkerSide(int dsize) {
    int server_tree = 0;
    std::cout << "Worker Starts\n" << std::endl;
    int sample_sz = dsize * conf.tree_sample;
    DataVector sample;
    sample.reserve(sample_sz);

    while (!server_finish_) {
      // get read lock to access data ptr
      data_ptr_lock_.RLock();
      std::sample(data_ptr_->begin(), data_ptr_->end(),
                  std::back_inserter(sample),
                  sample_sz, std::mt19937{std::random_device{}()});
      data_ptr_lock_.RUnlock();
      RegressionTree *iter_tree = new RegressionTree(conf);
      iter_tree->Fit(&sample, sample_sz);
      trees_vec_.push_and_notify(iter_tree);
      sample.clear();
      server_tree++;
    }
    std::cout << "Worker joins, trees generated: " << server_tree << std::endl;
  }

/*
 * temp_pred: cumulative predictions; already init outside
 * Pull a new tree if non-empty
 * Update the private data_vec (L)
 * once finish, acquire write lock and swap pointers
 */
  void GBDT::ServerSide(int dsize, std::vector <ValueType> &temp_pred) {
    std::cout << "Server Starts\n" << std::endl;
    int update_count = 0;
    assert(!server_finish_);
    while (update_count < conf.num_trees) {
      Elapsed elapsed;
      RegressionTree *new_tree = trees_vec_.wait_and_consume();
      data_ptr_lock_.WLock();
#pragma omp parallel for default(none) shared(dsize, temp_pred, data_ptr_, new_tree, conf) schedule(dynamic)
      for (int j = 0; j < dsize; ++j) {
        temp_pred[j] = PredictAsync(*(data_ptr_->at(j)), new_tree, temp_pred[j]);
        conf.loss->UpdateGradient(data_ptr_->at(j), temp_pred[j]);
      }
      update_count += 1;
      long fitting_time = elapsed.Tell().ToMilliseconds();
      if (conf.debug) {
        if (update_count % PRINT_TREE_INTERVAL == 0) {
          std::cout << "iteration: " << update_count << ", time: " << fitting_time << " milliseconds" << ", loss: "
          << GetLossSimple(data_ptr_, data_ptr_->size(), temp_pred) << std::endl;
        }
      }
      data_ptr_lock_.WUnlock();
    }
    server_finish_ = true;
  }


/*
 * Run the server routine inside the function call
 * And launch a vector of work threads
 * Join worker threads when the server finishes
 * (e.g. update for certain amount of trees)
 */
  void GBDT::Fit_Async(DataVector *d) {
    ReleaseTrees();
    size_t dsize = d->size();
    bias = conf.loss->GetBias(*d, dsize);
    // only for server: store temp value of pred for all data points
    std::vector <ValueType> temp_pred(dsize, bias);
    // ptr protected by RW lock
    data_ptr_ = d;
    std::cout << "Async start initialization..\n" << std::endl;
    // init target and grad for Datavector
    for (size_t j = 0; j < dsize; ++j) {
      temp_pred[j] = Predict_OMP(*(d->at(j)), 0, temp_pred[j]);
      conf.loss->UpdateGradient(d->at(j), temp_pred[j]);
    }
    std::cout << "Start launching threads..\n" << std::endl;
    // launch threads
    std::vector <std::thread> workers;
    workers.reserve(conf.num_of_threads - 1);
    for (int wt = 0; wt < conf.num_of_threads - 1; wt++) {
      workers.push_back(std::thread([=] { this->WorkerSide(dsize); }));
    }
    ServerSide(dsize, temp_pred);
    for (int i = 0; i < conf.num_of_threads - 1; i++) {
      workers[i].join();
    }
    data_ptr_ = nullptr;
    server_finish_ = false;

    // Calculate gain
    std::cout << "Processed trees in total: "  << conf.num_trees <<
    " total generated trees: " << trees_vec_.get_total_num() << std::endl;
    assert(trees_vec_.get_processed() >= conf.num_trees);
    std::cout << "Calculate gains...\n" << std::endl;

    gain = new double[conf.number_of_feature];
    for (int i = 0; i < conf.number_of_feature; ++i) {
      gain[i] = 0.0;
    }
    for (int j = 0; j < trees_vec_.get_processed(); ++j) {
      double *g = trees_vec_.get_elem(j)->GetGain();
      for (size_t i = 0; i < conf.number_of_feature; ++i) {
        gain[i] += g[i];
      }
    }
  }

  void GBDT::Fit_OMP(DataVector *d) {
    ReleaseTrees();
    size_t dsize = d->size();
    Init(*d);
    size_t sample_sz = static_cast<size_t>(dsize * conf.data_sample_ratio);
    // store temp value of pred for all data points
    std::vector <ValueType> temp_pred(dsize, bias);
    // assume in each iteration, num_of_threads number of trees will be built
    int iterations = (conf.num_trees - 1) / conf.num_of_threads + 1;

    for (size_t i = 0; i < iterations; ++i) {
      Elapsed elapsed;
      // update gradients for ALL data points
      // update cumulative pred and target field in tuples
#pragma omp parallel for default(none) shared(trees, d, dsize, i, conf, temp_pred) schedule(dynamic)
      for (int j = 0; j < dsize; ++j) {
        if (i > 0) {
          temp_pred[j] = Predict_OMP(*(d->at(j)), i, temp_pred[j]);
        }
        conf.loss->UpdateGradient(d->at(j), temp_pred[j]);
      }
      std::cout << "Finish updating for " << i << std::endl;

      int num_iter_tree = conf.num_of_threads;
      if ((i+1) * conf.num_of_threads > conf.num_trees) {
        num_iter_tree = conf.num_trees - i * conf.num_of_threads;
      }

      // build trees independently
#pragma omp parallel for default(none) shared(trees, d, dsize, sample_sz, i, num_iter_tree) schedule(dynamic)
      for (int j = 0; j < num_iter_tree; ++j) {
        // take a random sample
        DataVector sample;
        sample.reserve(sample_sz);
        // needs c++ 17
        std::sample(d->begin(), d->end(),
                    std::back_inserter(sample),
                    sample_sz, std::mt19937{std::random_device{}()});
        RegressionTree *iter_tree = trees[i * conf.num_of_threads + j];
        // fit a new tree based on updated target of tuples
        iter_tree->Fit(&sample, sample_sz);
      }
      std::cout << "Finish building trees for " << i << std::endl;

      long fitting_time = elapsed.Tell().ToMilliseconds();
      if (conf.debug) {
        std::cout << "iteration: " << i << ", time: " << fitting_time << " milliseconds"
                  << ", loss: " << GetLoss(d, d->size(), i, temp_pred) << std::endl;
      }
    }

    // Calculate gain
    delete[] gain;
    gain = new double[conf.number_of_feature];

    for (size_t i = 0; i < conf.number_of_feature; ++i) {
      gain[i] = 0.0;
    }

    for (size_t j = 0; j < conf.num_trees; ++j) {
      double *g = trees[j]->GetGain();
      for (size_t i = 0; i < conf.number_of_feature; ++i) {
        gain[i] += g[i];
      }
    }
  }

  std::string GBDT::Save(bool if_async) {
    std::vector <std::string> vs;
    vs.push_back(std::to_string(shrinkage));
    vs.push_back(std::to_string(bias));
    for (size_t i = 0; i < conf.num_trees; ++i) {
      if (if_async) {
        RegressionTree* t = trees_vec_.get_elem(i);
        vs.push_back(t->Save());
      } else {
        vs.push_back(trees[i]->Save());
      }
    }
    return JoinString(vs, "\n;\n");
  }

  void GBDT::Load(const std::string &s) {
    delete[] trees;
    std::vector <std::string> vs;
    SplitString(s, "\n;\n", &vs);

    shrinkage = std::stod(vs[0]);
    bias = std::stod(vs[1]);

    ReleaseTrees();

    conf.iterations = vs.size() - 2;
    trees = new RegressionTree *[conf.iterations];
    for (int i = 0; i < conf.iterations; ++i) {
      trees[i] = new RegressionTree(conf);
    }
    for (size_t i = 0; i < conf.iterations; ++i) {
      trees[i]->Load(vs[i + 2]);
    }
  }

  double GBDT::GetLoss(DataVector *d, size_t samples, int i, std::vector<ValueType> temp_pred) {
    double s = 0.0;
#ifdef USE_OPENMP
#pragma omp parallel for reduction(+:s)
#endif
    for (size_t j = 0; j < samples; ++j) {
      ValueType p = Predict_OMP(*(d->at(j)), i, temp_pred[j]);
      s += conf.loss->GetLoss(*(d->at(j)), p);
    }
    return s / samples;
  }

  double GBDT::GetLossSimple(DataVector *d, size_t samples, std::vector<ValueType> temp_pred) {
    double s = 0.0;
#ifdef USE_OPENMP
#pragma omp parallel for reduction(+:s)
#endif
    for (size_t j = 0; j < samples; ++j) {
      s += conf.loss->GetLoss(*(d->at(j)), temp_pred[j]);
    }
    return s / samples;
  }
}

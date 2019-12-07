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


#define NUM_INDEP_TREES 8
#define TREE_SAMPLE_THRESHOLD 0.3

namespace gbdt {

  void GBDT::Init(DataVector &d) {
    if (conf.enable_initial_guess) {
      return;
    }
    // this computes the weighted mean as init guess
    bias = conf.loss->GetBias(d, d.size());
    trees = new RegressionTree *[conf.iterations * NUM_INDEP_TREES];
    for (int i = 0; i < conf.iterations * NUM_INDEP_TREES; ++i) {
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

    for (size_t i = 0; i < iterations * NUM_INDEP_TREES; ++i) {
      r += shrinkage * trees[i]->Predict(t) / NUM_INDEP_TREES;
    }

    return r;
  }

  ValueType GBDT::PredictAsync(const Tuple &t, RegressionTree *tree, ValueType temp_pred) const {
    temp_pred += shrinkage / NUM_INDEP_TREES * tree->Predict(t);
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
    assert(n <= iterations);

    if (n == 0) {
      if (conf.enable_initial_guess) {
        return t.initial_guess;
      }
      return bias;
    }

    for (size_t i = (n - 1) * NUM_INDEP_TREES; i < n * NUM_INDEP_TREES; ++i) {
      temp_pred += shrinkage / NUM_INDEP_TREES * trees[i]->Predict(t);
    }

    return temp_pred;
  }

  /*
   * Generate a random sample of index first
   * Take read lock and fit a new tree
   * push it to the forest vector
   */
  void GBDT::WorkerSide(int dsize) {
    std::cout << "Worker Starts\n" << std::endl;
    int sample_sz = dsize * TREE_SAMPLE_THRESHOLD;
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
    }
    std::cout << "Worker joins\n" << std::endl;
  }

/*
 * temp_pred: cumulative predictions; already init outside
 * Pull a new tree if non-empty
 * Update the private data_vec (L)
 * once finish, acquire write lock and swap pointers
 */
  void GBDT::ServerSide(int dsize, int num_iter, std::vector <ValueType> &temp_pred) {
    std::cout << "Server Starts\n" << std::endl;
    int update_count = 0;
    assert(!server_finish_);
    while (update_count < num_iter) {
      Elapsed elapsed;
      RegressionTree *new_tree = trees_vec_.wait_and_consume();
      data_ptr_lock_.WLock();
      for (int j = 0; j < dsize; ++j) {
        temp_pred[j] = PredictAsync(*(data_ptr_->at(j)), new_tree, temp_pred[j]);
        conf.loss->UpdateGradient(data_ptr_->at(j), temp_pred[j]);
      }
      data_ptr_lock_.WUnlock();
      update_count += 1;
      long fitting_time = elapsed.Tell().ToMilliseconds();
      if (conf.debug) {
        std::cout << "iteration: " << update_count << ", time: " << fitting_time << " milliseconds\n"
                  << std::endl;
      }
    }
    server_finish_ = true;
  }


/*
 * Run the server routine inside the function call
 * And launch a vector of work threads
 * Join worker threads when the server finishes
 * (e.g. update for certain amount of trees)
 */
  void GBDT::Fit_Async(DataVector *d, int threads_wanted) {
    ReleaseTrees();
    size_t dsize = d->size();
    bias = conf.loss->GetBias(*d, dsize);
    // only for server: store temp value of pred for all data points
    std::vector <ValueType> temp_pred(dsize, bias);
    // ptr protected by RW lock
    data_ptr_ = d;
    // init target and grad for Datavector
    for (size_t j = 0; j < dsize; ++j) {
      temp_pred[j] = Predict_OMP(*(d->at(j)), 0, temp_pred[j]);
      conf.loss->UpdateGradient(d->at(j), temp_pred[j]);
    }
    std::cout << "Start launching threads..\n" << std::endl;
    // launch threads
    std::vector <std::thread> workers;
    workers.reserve(threads_wanted - 1);
    for (int wt = 0; wt < threads_wanted - 1; wt++) {
      workers.push_back(std::thread([=] { this->WorkerSide(dsize); }));
    }
    ServerSide(dsize, iterations * NUM_INDEP_TREES, temp_pred);
    for (int i = 0; i < threads_wanted - 1; i++) {
      workers[i].join();
    }
    data_ptr_ = nullptr;
    server_finish_ = false;

    // Calculate gain
    std::cout << "Processed trees in total: " << trees_vec_.get_processed() <<
    " should be: " << iterations * NUM_INDEP_TREES << std::endl;
    assert(trees_vec_.get_processed() >= iterations * NUM_INDEP_TREES);
    std::cout << "Calculate gains...\n" << std::endl;

    gain = new double[conf.number_of_feature];
    for (int i = 0; i < conf.number_of_feature; ++i) {
      gain[i] = 0.0;
    }
    for (int j = 0; j < iterations * NUM_INDEP_TREES; ++j) {
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

      // build trees independently
#pragma omp parallel for default(none) shared(trees, d, dsize, sample_sz, i) schedule(dynamic)
      for (int j = 0; j < NUM_INDEP_TREES; ++j) {
        // take a random sample
        DataVector sample;
        sample.reserve(dsize);
        // needs c++ 17
        std::sample(d->begin(), d->end(),
                    std::back_inserter(sample),
                    sample_sz, std::mt19937{std::random_device{}()});
        RegressionTree *iter_tree = trees[i * NUM_INDEP_TREES + j];
        // fit a new tree based on updated target of tuples
        iter_tree->Fit(&sample, sample_sz);
      }

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

    for (size_t j = 0; j < iterations * NUM_INDEP_TREES; ++j) {
      double *g = trees[j]->GetGain();
      for (size_t i = 0; i < conf.number_of_feature; ++i) {
        gain[i] += g[i];
      }
    }
  }

  std::string GBDT::Save(bool if_async) const {
    std::vector <std::string> vs;
    vs.push_back(std::to_string(shrinkage));
    vs.push_back(std::to_string(bias));
    for (size_t i = 0; i < iterations; ++i) {
      if (if_async) {
        vs.push_back(trees_vec_.get_elem(i)->Save());
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

    iterations = vs.size() - 2;
    shrinkage = std::stod(vs[0]);
    bias = std::stod(vs[1]);

    ReleaseTrees();

    conf.iterations = iterations;
    trees = new RegressionTree *[iterations];
    for (int i = 0; i < iterations; ++i) {
      trees[i] = new RegressionTree(conf);
    }
    for (size_t i = 0; i < iterations; ++i) {
      trees[i]->Load(vs[i + 2]);
    }
  }

  GBDT::~GBDT() {
    ReleaseTrees();
    trees_vec_.destroy_all();
    delete[] gain;
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
}

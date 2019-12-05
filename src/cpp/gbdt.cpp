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

  void GBDT::ServerSide(std::queue<RegressionTree *> &taskQ, ValueType *pub_ptr, int dsize) {
    ValueType private_L[dsize];
    ValueType tempL = *pub_ptr;
    std::copy(std::begin(tempL), std::end(tempL), std::begin(private_L));
    ValueType *priv_ptr;
    priv_ptr = private_L;

    while (true) {
      while (taskQ.empty()) {
        //wait for a new tree to be made
      }
      //under lock
      RegressionTree curr_tree = taskQ.pop();

      trees.push_back(curr_tree);
      //Compute the updated L
      for (int j = 0; j < dsize; ++j) {
        if (i > 0) {
          *priv_ptr[j] = Predict_OMP(*(d->at(j)), i, *pub_ptr[j]);
        }
        conf.loss->UpdateGradient(d->at(j), *priv_ptr[j]);
      }
      //under lock
      ValueType *temp_ptr = priv_ptr;
      priv_ptr = pub_ptr;
      pub_ptr = temp_ptr;
    }

  }

  void GBDT::WorkerSide(std::queue<RegressionTree *> &taskQ,, ValueType

  * pub_ptr) {

}

void GBDT::Fit(DataVector *d, int threads_wanted) {
  ReleaseTrees();
  size_t dsize = d->size();
  Init(*d);
  size_t sample_sz = static_cast<size_t>(dsize * conf.data_sample_ratio);
  // store temp value of pred for all data points
  ValueType temp_pred[dsize] = {bias};
  //Create a task queue
  queue <RegressionTree> taskQ;
  //Create a pointer to the public L
  ValueType *pub_ptr;
  pub_ptr = temp_pred;

  std::thread server(ServerSide, &taskQ, pub_ptr, dsize);
  std::vector <std::thread> workers;
  for (int wt = 1; wt < threads_wanted; wt++) {
    workers.push_back(std::thread(WorkerSide, &taskQ, pub_ptr));
  }
  for (size_t i = 0; i < conf.iterations; ++i) {
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

std::string GBDT::Save() const {
  std::vector <std::string> vs;
  vs.push_back(std::to_string(shrinkage));
  vs.push_back(std::to_string(bias));
  for (size_t i = 0; i < iterations; ++i) {
    vs.push_back(trees[i]->Save());
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
  delete[] gain;
}

double GBDT::GetLoss(DataVector *d, size_t samples, int i, ValueType *temp_pred) {
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

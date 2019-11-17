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

#ifdef USE_OPENMP
#include <parallel/algorithm>  // openmp
#endif

#define NUM_INDEP_TREES 8
#define TREE_SAMPLE_THRESHOLD 0.3

namespace gbdt {
  ValueType GBDT::Predict(const Tuple &t, size_t n) const {
    if (!trees)
      return kUnknownValue;

    assert(n <= iterations);

    ValueType r = bias;
    if (conf.enable_initial_guess) {
      r = t.initial_guess;
    }

    for (size_t i = 0; i < n; ++i) {
      r += shrinkage * trees[i]->Predict(t);
    }

    return r;
  }

  ValueType GBDT::Predict(const Tuple &t, size_t n, double *p) const {
    if (!trees)
      return kUnknownValue;

    assert(n <= iterations);

    ValueType r = bias;
    if (conf.enable_initial_guess) {
      r = t.initial_guess;
    }

    for (size_t i = 0; i < n; ++i) {
      r += shrinkage * trees[i]->Predict(t, p);
    }

    return r;
  }

  void GBDT::Init(DataVector &d, size_t len) {
    assert(d.size() >= len);

    if (conf.enable_initial_guess) {
      return;
    }

    bias = conf.loss->GetBias(d, len);

    trees = new RegressionTree *[conf.iterations];
    for (int i = 0; i < conf.iterations; ++i) {
      trees[i] = new RegressionTree(conf);
    }

  }

  void GBDT::Fit(DataVector *d) {
    ReleaseTrees();
    size_t dsize = d->size();
    Init(*d, dsize * NUM_INDEP_TREES);
    size_t sample_sz = static_cast<size_t>(dsize * conf.data_sample_ratio);
    // presort only once
    std::random_shuffle(d->begin(), d->end());

    for (size_t i = 0; i < conf.iterations; ++i) {
      Elapsed elapsed;

#pragma omp parallel for default(none) shared(trees, d, samples, i) schdule(static)
      for (int j = 0; j < NUM_INDEP_TREES; ++j) {
        // take a random sample
        std::vector<int> sample;
        std::sample(population.begin(), population.end(),
                    std::back_inserter(sample),
                    sample_sz, std::mt19937{std::random_device{}()});
        RegressionTree* iter_tree = trees[i * NUM_INDEP_TREES + j];
        iter_tree->Fit(sample, sample_sz);
      }

      UpdateGradient(d, samples, i);
      trees[i]->Fit(d, samples);
      long fitting_time = elapsed.Tell().ToMilliseconds();
      if (conf.debug) {
        std::cout << "iteration: " << i << ", time: " << fitting_time << " milliseconds"
                  << ", loss: " << GetLoss(d, samples, i) << std::endl;
      }
    }


    // Calculate gain
    delete[] gain;
    gain = new double[conf.number_of_feature];

    for (size_t i = 0; i < conf.number_of_feature; ++i) {
      gain[i] = 0.0;
    }

    for (size_t j = 0; j < iterations; ++j) {
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

  void GBDT::UpdateGradient(DataVector *d, size_t samples, int i) {
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (size_t j = 0; j < samples; ++j) {
      ValueType p = Predict(*(d->at(j)), i);
//    ???? no sync???
      conf.loss->UpdateGradient(d->at(j), p);
    }
  }

  double GBDT::GetLoss(DataVector *d, size_t samples, int i) {
    double s = 0.0;
#ifdef USE_OPENMP
#pragma omp parallel for reduction(+:s)
#endif
    for (size_t j = 0; j < samples; ++j) {
      ValueType p = Predict(*(d->at(j)), i);
      s += conf.loss->GetLoss(*(d->at(j)), p);
    }

    return s / samples;
  }

}

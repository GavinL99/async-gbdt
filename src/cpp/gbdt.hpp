// Author: qiyiping@gmail.com (Yiping Qi)
#ifndef _GBDT_H_
#define _GBDT_H_
#include "tree_seq.hpp"

namespace gbdt {
class GBDT {
 public:
  GBDT(Configure conf): trees(NULL),
                        bias(0),
                        conf(conf),
                        gain(NULL) {
    shrinkage = conf.shrinkage;
    iterations = conf.iterations;
  }

  void Fit(DataVector *d);
  ValueType Predict(const Tuple &t) const {
    return Predict(t, iterations);
  }

  std::string Save() const;
  void Load(const std::string &s);

  double *GetGain() { return gain; }

  ~GBDT();
 private:
  void Init(DataVector &d, size_t len);

  void UpdateGradient(DataVector *d, size_t samples, int iteration);
  double GetLoss(DataVector *d, size_t samples, int i);

  ValueType GBDT::Predict_OMP(const Tuple &t, size_t n, ValueType temp_pred) const;

  void ReleaseTrees() {
    if (trees) {
      for (int i = 0; i < iterations; ++i) {
        delete trees[i];
      }
      delete[] trees;
      trees = NULL;
    }
  }

 private:
  RegressionTree **trees;
//  // for trees
//  double *weights;
  ValueType bias;
  ValueType shrinkage;
  size_t iterations;

  Configure conf;

  double *gain;

  DISALLOW_COPY_AND_ASSIGN(GBDT);
};
}

#endif /* _GBDT_H_ */

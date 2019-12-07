// Author: qiyiping@gmail.com (Yiping Qi)

#include "gbdt.hpp"
#include <iostream>
#include <fstream>
#include <cassert>
#include "time.hpp"
#include "cmd_option.hpp"
#include "loss.hpp"
#include "common_loss.hpp"
#include <queue> 


using namespace gbdt;

Objective* create__SquaredError() { return new SquaredError(); }

int main(int argc, char *argv[]) {
  CmdOption opt;
  opt.AddOption("threads", "t", "threads", 1);
  opt.AddOption("feature_size", "f", "feature_size", OptionType::INT, true);
  opt.AddOption("max_depth", "d", "max_depth", 4);
  opt.AddOption("iterations", "n", "iterations", 10);
  opt.AddOption("shrinkage", "s", "shrinkage", 0.1);
  opt.AddOption("feature_ratio", "r", "feature_ratio", 1.0);
  opt.AddOption("data_ratio", "R", "data_ratio", 1.0);
  opt.AddOption("debug", "D", "debug", false);
  opt.AddOption("min_leaf_size", "S", "min_leaf_size", 0);
  opt.AddOption("loss", "l", "loss", "SquaredError");
  opt.AddOption("train_file", "F", "train_file", OptionType::STRING, true);
  opt.AddOption("custom_loss_so", "c", "custom_loss_so", "");

  if (!opt.ParseOptions(argc, argv)) {
    opt.Help();
    return -1;
  }

  int threads_wanted = 16;
  opt.Get("num_of_threads", &threads_wanted);
  std::srand ( unsigned ( ::time(0) ) );

  Configure conf;
  opt.Get("feature_size", &conf.number_of_feature);
  opt.Get("max_depth", &conf.max_depth);
  opt.Get("iterations", &conf.iterations);
  opt.Get("shrinkage", &conf.shrinkage);
  opt.Get("feature_ratio", &conf.feature_sample_ratio);
  opt.Get("data_ratio", &conf.data_sample_ratio);
  opt.Get("debug", &conf.debug);
  opt.Get("min_leaf_size", &conf.min_leaf_size);
//  std::string loss_type;
//  opt.Get("loss", &loss_type);
  std::string loss_type = "SquaredError";

  std::string custom_loss_so;
  opt.Get("custom_loss_so", &custom_loss_so);
  LossFactory::GetInstance()->LoadSharedLib(custom_loss_so);

  std::cout << "Start registering objective function\n" << std::endl;
  using CreateFn = Objective* (*) ();
  LossFactory::GetInstance()->Register("SquaredError", create__SquaredError);        \
  Objective *objective = LossFactory::GetInstance()->Create(loss_type);
  if (!objective) {
    LossFactory::GetInstance()->PrintAllCandidates();
    return -1;
  }
  std::cout << "Objective function registered!\n" << std::endl;
  conf.loss.reset(objective);

  std::string train_file;
  opt.Get("train_file", &train_file);
  DataVector d;
  bool r = LoadDataFromFile(train_file,
                            &d,
                            conf.number_of_feature,
                            loss_type == "LogLoss");
  assert(r);

  GBDT gbdt(conf);

  Elapsed elapsed;

//  gbdt.Fit(&d,threads_wanted);
  std::cout << "Start training..\n" << std::endl;
  gbdt.Fit_Async(&d,threads_wanted);
  std::cout << "training time: " << elapsed.Tell().ToMilliseconds() << " milliseconds" << std::endl;
  CleanDataVector(&d);
  FreeVector(&d);
  std::cout << "Clean up...\n";

  std::string model_file = train_file + ".model";
  std::ofstream model_output(model_file.c_str());
  model_output << gbdt.Save(true);
  std::cout << "Model saved...\n";
  double *g = gbdt.GetGain();
  std::cout << "feature index\tfeature gain" << std::endl;
  for (size_t i = 0; i < conf.number_of_feature; ++i) {
    std::cout << i << "\t" << g[i] << std::endl;
  }

  return 0;
}

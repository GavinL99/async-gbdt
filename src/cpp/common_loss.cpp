#include "common_loss.hpp"

namespace gbdt {
  using CreateFn = Objective* (*) ();

//DEFINE_OBJECTIVE_REGISTRATION(SquaredError)
//
//DEFINE_OBJECTIVE_REGISTRATION(LogLoss)
//
//DEFINE_OBJECTIVE_REGISTRATION(LAD)

//void register__SquaredError(void) {                                        \
//  LossFactory::GetInstance()->Register("SquaredError", (CreateFn) new SquaredError());        \
//}
//
//void register__LogLoss(void) {                                        \
//  LossFactory::GetInstance()->Register("LogLoss", (CreateFn) new LogLoss());        \
//}
//
//void register__LAD(void) {                                        \
//  LossFactory::GetInstance()->Register("LAD", (CreateFn) new LAD());        \
//}

}  // gbdt

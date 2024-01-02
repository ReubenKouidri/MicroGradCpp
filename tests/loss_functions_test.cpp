#include <gtest/gtest.h>
#include "../include/components.hpp"
#include "../include/losses.hpp"
#include "../include/optimiser.hpp"
#include "trainer.hpp"

class LossFunctionsTest : public testing::Test {
 protected:
  void SetUp() override {
  }

  const std::vector<double> input0{1.0, 0.0, 0.0};
  const std::vector<double> input1{0.0, 1.0, 0.0};
  const std::vector<double> input2{0.0, 0.0, 1.0};

  static constexpr uint8_t tgt0 = 0;
  static constexpr uint8_t tgt1 = 1;
  static constexpr uint8_t tgt2 = 2;

  std::vector<uint8_t> batched_sparse_tgt{tgt0, tgt1, tgt2};

  const std::vector<uint8_t> categorical_tgt0{1, 0, 0};
  const std::vector<uint8_t> categorical_tgt1{0, 1, 0};
  const std::vector<uint8_t> categorical_tgt2{0, 0, 1};

  const std::vector<std::vector<uint8_t>> batched_categorical_tgt{
      categorical_tgt0,
      categorical_tgt1,
      categorical_tgt2};

  std::vector<std::vector<double>> batched_input{input0, input1, input2};

  static constexpr double learning_rate = 0.05;
  static constexpr size_t epochs = 25;

  const MLP<double> model{{
                              Layer<double>(3, 3, UnaryOp::relu),
                              Layer<double>(3, 3, UnaryOp::softmax)
                          }};
  const std::shared_ptr<const MLP<double>>
      mp = std::make_shared<MLP<double>>(model);

  SparseCCELoss<double> sparse_cce_loss{mp};
  Adam<double> adam{mp, learning_rate};
  CCELoss<double> cce_loss{mp};
  MSELoss<double> mse_loss{mp};
};

TEST_F(LossFunctionsTest, test_sparse_cce) {
  std::cout << "========================================\n";
  std::cout << "======= TESTING SPARSE CCE LOSS ========\n";
  std::cout << "========================================\n";

  train_model(mp,
              input0,
              tgt0,
              sparse_cce_loss,
              adam,
              epochs);
}

TEST_F(LossFunctionsTest, test_sparse_cce_batched) {
  std::cout << "==============================================\n";
  std::cout << "=======TESTING BATCHED SPARSE CCE LOSS========\n";
  std::cout << "==============================================\n";
  train_model(mp,
              batched_input,
              batched_sparse_tgt,
              sparse_cce_loss,
              adam,
              epochs);
}

TEST_F(LossFunctionsTest, test_cce) {
  std::cout << "============================================\n";
  std::cout << "============= TESTING CCE LOSS =============\n";
  std::cout << "============================================\n";
  train_model(mp,
              input0,
              categorical_tgt0,
              cce_loss,
              adam,
              epochs);
}

TEST_F(LossFunctionsTest, test_batched_cce) {
  std::cout << "=============================================\n";
  std::cout << "========= TESTING BATCHED CCE LOSS ==========\n";
  std::cout << "=============================================\n";
  train_model(mp,
              batched_input,
              batched_categorical_tgt,
              cce_loss,
              adam,
              epochs);
}

TEST_F(LossFunctionsTest, test_mse) {
  std::cout << "======================================\n";
  std::cout << "==========TESTING MSE LOSS============\n";
  std::cout << "======================================\n";
  train_model(mp,
              input0,
              tgt0,
              mse_loss,
              adam,
              epochs);
}

TEST_F(LossFunctionsTest, test_batched_mse) {
  std::cout << "==============================================\n";
  std::cout << "==========TESTING BATCHED MSE LOSS============\n";
  std::cout << "==============================================\n";
  train_model(mp,
              batched_input,
              batched_sparse_tgt,
              mse_loss,
              adam,
              epochs);
}
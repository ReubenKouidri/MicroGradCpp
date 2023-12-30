#include <gtest/gtest.h>
#include "../include/components.hpp"
#include "../include/losses.hpp"
#include "../include/optimiser.hpp"

class LossFunctionsTest : public testing::Test {
protected:
  void SetUp() override {
  }

  const std::vector<double> input {1.0, 0.0, 1.0};
  static constexpr uint8_t tgt = 0;

  const std::vector<uint8_t> categorical_tgt {1, 0};
  const std::vector<std::vector<uint8_t>> batched_categorical_tgt {
    categorical_tgt, categorical_tgt, categorical_tgt};

  std::vector<std::vector<double>> batched_input {input, input, input};
  std::vector<uint8_t> batched_tgt {tgt, tgt, tgt};

  static constexpr double learning_rate = 0.05;
  static constexpr size_t epochs = 25;

  const Layer<double> layer0 {3, 3, UnaryOp::relu};
  const Layer<double> layer1 {3, 2, UnaryOp::softmax};
  MLP<double> model {{layer0, layer1}};
  SparseCCELoss<double> sparse_cce_loss {model, learning_rate};
  CCELoss<double> cce_loss {model, learning_rate};
  MSELoss<double> mse_loss {model, learning_rate};
};

TEST_F(LossFunctionsTest, test_sparse_cce) {
  std::cout << "========================================\n";
  std::cout << "======= TESTING SPARSE CCE LOSS ========\n";
  std::cout << "========================================\n";
  train_model(model,
                input,
                tgt,
                sparse_cce_loss,
                learning_rate,
                epochs);
}

TEST_F(LossFunctionsTest, test_sparse_cce_batched) {
  std::cout << "==============================================\n";
  std::cout << "=======TESTING BATCHED SPARSE CCE LOSS========\n";
  std::cout << "==============================================\n";
  train_model(model,
                batched_input,
                batched_tgt,
                sparse_cce_loss,
                learning_rate,
                epochs);
}

TEST_F(LossFunctionsTest, test_cce) {
  std::cout << "============================================\n";
  std::cout << "============= TESTING CCE LOSS =============\n";
  std::cout << "============================================\n";
  train_model(model,
              input,
              categorical_tgt,
              cce_loss,
              learning_rate,
              epochs);
}

TEST_F(LossFunctionsTest, test_batched_cce) {
  std::cout << "=============================================\n";
  std::cout << "========= TESTING BATCHED CCE LOSS ==========\n";
  std::cout << "=============================================\n";
  train_model(model,
                 batched_input,
                 batched_categorical_tgt,
                 cce_loss,
                 learning_rate,
                 epochs);
}

TEST_F(LossFunctionsTest, test_mse) {
  std::cout << "======================================\n";
  std::cout << "==========TESTING MSE LOSS============\n";
  std::cout << "======================================\n";
  train_model(model,
                input,
                tgt,
                mse_loss,
                learning_rate,
                epochs);
}

TEST_F(LossFunctionsTest, test_batched_mse) {
  std::cout << "==============================================\n";
  std::cout << "==========TESTING BATCHED MSE LOSS============\n";
  std::cout << "==============================================\n";
  train_model(model,
                batched_input,
                batched_tgt,
                mse_loss,
                learning_rate,
                epochs);
}
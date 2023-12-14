#include <gtest/gtest.h>
#include "../include/module.hpp"
#include "../include/losses.hpp"

class LossFunctionsTest : public testing::Test {
protected:
  void SetUp() override {
  }

  template<class Loss, class Input_Tp, class Target_Tp>
  void train_model(const Input_Tp& input,
                   const Target_Tp& target,
                   Loss& loss_function,
                   const size_t epochs)
  {
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
      auto loss = loss_function.compute_loss(input, target);
      loss_function.zero_grad();
      loss.backward();
      loss_function.step();
      std::cout << "Loss = " << loss << '\n';
    }
  }

  const std::vector<size_t> neurons {3, 3, 2};
  const std::vector<double> input {1.0, 0.0, 1.0};
  static constexpr uint8_t tgt = 0;

  const std::vector<uint8_t> categorical_tgt {1, 0};
  const std::vector<std::vector<uint8_t>> batched_categorical_tgt {categorical_tgt, categorical_tgt, categorical_tgt};

  std::vector<std::vector<double>> batched_input {input, input, input};
  std::vector<uint8_t> batched_tgt {tgt, tgt, tgt};

  static constexpr double learning_rate = 0.05;
  static constexpr size_t epochs = 100;

  const Layer<double> layer0 {3, 3, Activation::RELU};
  const Layer<double> layer1 {3, 2, Activation::SOFTMAX};
  MLP<double> model { {layer0, layer1} };
  SparseCCELoss<double> sparse_cce_loss {model, learning_rate};
  CCELoss<double> cce_loss {model, learning_rate};
  MSELoss<double> mse_loss {model, learning_rate};
};

TEST_F(LossFunctionsTest, test_sparse_cce) {
  std::cout << "========================================\n";
  std::cout << "======= TESTING SPARSE CCE LOSS ========\n";
  std::cout << "========================================\n";
  train_model<decltype(sparse_cce_loss),
              decltype(input),
              decltype(tgt)>
  (input, tgt, sparse_cce_loss, 100);
}

TEST_F(LossFunctionsTest, test_sparse_cce_batched) {
  std::cout << "==============================================\n";
  std::cout << "=======TESTING BATCHED SPARSE CCE LOSS========\n";
  std::cout << "==============================================\n";
  train_model<decltype(sparse_cce_loss),
              decltype(batched_input),
              decltype(batched_tgt)>
  (batched_input, batched_tgt, sparse_cce_loss, 100);
}

TEST_F(LossFunctionsTest, test_cce) {
  std::cout << "============================================\n";
  std::cout << "============= TESTING CCE LOSS =============\n";
  std::cout << "============================================\n";
  train_model<decltype(cce_loss),
              decltype(input),
              decltype(categorical_tgt)>
  (input, categorical_tgt, cce_loss, 100);
}

TEST_F(LossFunctionsTest, test_batched_cce) {
  std::cout << "=============================================\n";
  std::cout << "========= TESTING BATCHED CCE LOSS ==========\n";
  std::cout << "=============================================\n";
  train_model<decltype(cce_loss),
              decltype(batched_input),
              decltype(batched_categorical_tgt)>
  (batched_input, batched_categorical_tgt, cce_loss, 100);
}

TEST_F(LossFunctionsTest, test_mse) {
  std::cout << "======================================\n";
  std::cout << "==========TESTING MSE LOSS============\n";
  train_model<decltype(mse_loss),
              decltype(input),
              decltype(tgt)>
  (input, tgt, mse_loss, 100);
}

TEST_F(LossFunctionsTest, test_batched_mse) {
  std::cout << "==============================================\n";
  std::cout << "==========TESTING BATCHED MSE LOSS============\n";
  train_model<decltype(mse_loss),
              decltype(batched_input),
              decltype(batched_tgt)>
  (batched_input, batched_tgt, mse_loss, 100);
}
#include <gtest/gtest.h>
#include "../include/module.hpp"
#include "../include/losses.hpp"

class MLPTest : public testing::Test {
protected:
  void SetUp() override {
  }

  template<class T, class Loss, class Target_Tp>
  void train_model(const std::vector<T>& input,
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
  const std::vector<uint8_t> target {1, 0};  // for batch or ohe
  static constexpr uint8_t tgt = 0;
  static constexpr double learning_rate = 0.05;
  static constexpr size_t epochs = 100;

  const Layer<double> layer0 {3, 3, Activation::RELU};
  const Layer<double> layer1 {3, 2, Activation::SOFTMAX};
  MLP<double> model { {layer0, layer1} };
  SparseCCELoss<double, std::vector<double>, uint8_t> sparse_cce_loss {model, learning_rate};
  MSELoss<double, std::vector<double>, uint8_t> mse_loss {model, learning_rate};
};

TEST_F(MLPTest, test_sparse_cce) {
  std::cout << "======================================\n";
  std::cout << "=======TESTING SPARSE-CCE LOSS========\n";
  train_model<double, decltype(sparse_cce_loss)>(input, tgt, sparse_cce_loss, 100);
  std::cout << "======================================\n";
};

TEST_F(MLPTest, test_mse) {
  std::cout << "======================================\n";
  std::cout << "==========TESTING MSE LOSS============\n";
  train_model<double, decltype(mse_loss)>(input, tgt, mse_loss, 100);
  std::cout << "======================================\n";
}
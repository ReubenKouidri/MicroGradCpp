#include <gtest/gtest.h>
#include "../include/components.hpp"
#include "../include/losses.hpp"
#include "../include/optimiser.hpp"

template <typename T, class Loss, class Input_Tp, class Target_Tp>
inline void train(Adam<T> &optim,
                  const Input_Tp &inputs,
                  const Target_Tp &targets,
                  Loss &loss,
                  const size_t epochs) {
  const auto num_samples = inputs.size();
  for (size_t e = 0; e < epochs; e++) {
    double epoch_loss = 0;
    for (size_t i = 0; i < num_samples; i++) {
      loss.compute_loss(inputs, targets);
      epoch_loss += loss.get();
      loss.backward();
      optim.step();
      optim.zero_grad();
      loss.zero();
    }
    std::cout << "Epoch " << e << ": " << "Loss = " << epoch_loss << '\n';
  }
}

class AdamTest : public testing::Test {
 protected:
  void SetUp() override {
  }

  const std::vector<double> input{1.0, 0.0, 1.0};
  static constexpr uint8_t tgt = 0;

  const std::vector<uint8_t> categorical_tgt{1, 0};
  const std::vector<std::vector<uint8_t>> batched_categorical_tgt{
      categorical_tgt, categorical_tgt, categorical_tgt};

  std::vector<std::vector<double>> batched_input{input, input, input};
  std::vector<uint8_t> batched_tgt{tgt, tgt, tgt};

  static constexpr double learning_rate = 0.05;
  static constexpr size_t epochs = 25;

  const Layer<double> layer0{3, 3, UnaryOp::relu};
  const Layer<double> layer1{3, 2, UnaryOp::softmax};
  MLP<double> model{{layer0, layer1}};
  SparseCCELoss<double> sparse_cce_loss{&model};
  Adam<double> adam{&model, learning_rate};
};

TEST_F(AdamTest, test_adam) {
  std::cout << "========================================\n";
  std::cout << "============ TESTING ADAM -=============\n";
  std::cout << "========================================\n";
  train(adam,
        input,
        tgt,
        sparse_cce_loss,
        epochs);
}
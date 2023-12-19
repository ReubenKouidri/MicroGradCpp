#include <vector>
#include "include/module.hpp"
#include "include/losses.hpp"
#include "include/utils.hpp"

int main() {
  const std::vector<double> input {1.0, 0.0, 1.0};
  static constexpr uint8_t tgt = 0;

  const std::vector<uint8_t> categorical_tgt {1, 0};
  const std::vector<std::vector<uint8_t>> batched_categorical_tgt {categorical_tgt, categorical_tgt, categorical_tgt};

  const std::vector<decltype(input)> batched_input {input, input, input};
  const std::vector<decltype(tgt)> batched_target {tgt, tgt, tgt};

  constexpr auto learning_rate = 0.05;
  constexpr size_t epochs = 50;

  const Layer<double> layer0 {3, 3, UnaryOp::relu};
  const Layer<double> layer1 {3, 2, UnaryOp::softmax};
  MLP<double> model { {layer0, layer1} };
  SparseCCELoss<double> sparse_cce_loss {model, learning_rate};
  CCELoss<double> cce_loss {model, learning_rate};
  MSELoss<double> mse_loss {model, learning_rate};
  train_model(model, input, tgt, sparse_cce_loss, learning_rate, epochs);
}
#ifndef LOSSES_HPP
#define LOSSES_HPP

#include "value.hpp"
#include "module.hpp"

template<class T>
class MSELoss {
  MLP<T> network_;
  double learning_rate_{};
public:
  explicit MSELoss(const MLP<T>& network, const double learning_rate)
    : network_(network),
      learning_rate_(learning_rate) {
  }

  Value<T> compute_loss(const std::vector<Value<T>>& input, const std::vector<Value<T>>& targets) {
    auto output = network_(input);
    auto loss = Value<T>(0.0);
    for (size_t i = 0; i < output.size(); i++) {
      auto diff = output[i] - targets[i];
      auto temp = pow(diff, static_cast<T>(2));
      loss = loss + temp;
    }
    loss = loss / output.size();
    std::cout << "Output: " << output << "\nLoss: " << loss << "\n\n";
    return loss;
  }

  std::vector<Value<T>> convert(const std::vector<T>& input) {
    std::vector<Value<T>> new_inputs;
    new_inputs.reserve(input.size());
    for (const auto& num : input) {
      new_inputs.emplace_back(num);
    }
    return new_inputs;
  }

  Value<T> compute_loss(const std::vector<T>& input, const std::vector<T>& targets) {
    auto new_inputs = convert(input);
    auto new_targets = convert(targets);
    return compute_loss(new_inputs, new_targets);
  }

  void step() const { network_.step(learning_rate_); }
  void zero_grad() const { network_.zero_grad(); }
};

#endif //LOSSES_HPP

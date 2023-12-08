#ifndef LOSSES_HPP
#define LOSSES_HPP

#include "value.hpp"
#include "module.hpp"

template<class T>
class MSELoss {
  MLP<T> network_;
  std::vector<Value<T>> targets_;
  double learning_rate_{};
  Value<T> value_ = Value<T>(0.0);
public:
  explicit MSELoss(const MLP<T>& network, const std::vector<T>& targets, const double learning_rate)
    : MSELoss(network, convert_targets(targets), learning_rate) {
  }
  explicit MSELoss(const MLP<T>& network, const std::vector<Value<T>>& targets, const double learning_rate)
    : network_(network),
      targets_(targets),
      learning_rate_(learning_rate) {
  }
  std::vector<Value<T>> convert_targets(const std::vector<T>& targets) const {
    std::vector<Value<T>> new_targets;
    new_targets.reserve(targets.size());
    for (const auto& t : targets)
      new_targets.emplace_back(Value<T>(t));
    return new_targets;
  }
  void compute_loss(std::vector<Value<T>>& input) {
    auto output = network_(input);
    for (size_t i = 0; i < output.size(); i++) {
      auto diff = output[i] - targets_[i];
      auto temp = pow(diff, static_cast<T>(2));
      value_ = value_ + temp;
    }
    value_ = value_ / output.size();
  }

  void compute_loss(std::vector<T>& input) {
    std::vector<Value<T>> new_inputs;
    new_inputs.reserve(input.size());
    for (const auto& num : input) {
      new_inputs.emplace_back(num);
    }
    compute_loss(new_inputs);
  }
  const T& get() { return value_.get_data(); }
  void backward() { value_.backward(); }
  void step() const { network_.step(learning_rate_); }
  void zero_grad() const { network_.zero_grad(); }
};

#endif //LOSSES_HPP

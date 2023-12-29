#include "../include/layer.hpp"

template <typename T>
Layer<T>::Layer(const size_t nin, const size_t nout, const UnaryOp& activation)
    : activation_(activation) {
  neurons_.reserve(nout);
  for (size_t i = 0; i < nout; i++) {
    neurons_.emplace_back(Neuron<T>(nin, nout, activation));
  }
}

template <typename T>
Layer<T>::Layer(const Layer& other) : neurons_(other.neurons_),
                                      activation_(other.activation_) {}

template <typename T>
Layer<T>::Layer(Layer&& other) noexcept
    : neurons_(std::move(other.neurons_)),
      activation_(other.activation_) {}

template <typename T>
Layer<T>& Layer<T>::operator=(const Layer& other) {
  if (this != &other) {
    neurons_ = other.neurons_;
    activation_ = other.activation_;
  }
  return *this;
}

template <typename T>
Layer<T>& Layer<T>::operator=(Layer&& other) noexcept {
  if (this != &other) {
    neurons_ = std::move(other.neurons_);
    activation_ = other.activation_;
  }
  return *this;
}

template <typename T>
ParamVector<T> Layer<T>::get_parameters() const {
  ParamVector<T> params;
  for (const auto& n : neurons_) {
    auto temp = n.get_parameters();
    params.insert(params.end(), temp.begin(), temp.end());
  }
  return params;
}

template <typename T>
Output<T> Layer<T>::operator()(const std::vector<Value<T>>& inputs) const {
  Output<T> output;
  for (const auto& n : neurons_) {
    output.emplace_back(n(inputs));
  }
  if (activation_ == UnaryOp::softmax) {
    auto max_val = *std::max_element(
        output.begin(), output.end(),
        [&](const Value<T>& a, const Value<T>& b) { return a < b; });
    auto sum = Value(0.0);
    for (auto& o : output) {
      o = ops::exp(o - max_val);
      sum += o;
    }
    for (auto& o : output)
      o /= sum;
  }
  return output;
}

template <typename T>
Output<T> Layer<T>::operator()(const std::vector<T>& input) const {
  std::vector<Value<T>> new_input;
  new_input.reserve(input.size());
  for (const auto t : input) {
    new_input.emplace_back(Value(t));
  }
  return operator()(new_input);
}

template
class Layer<double>;

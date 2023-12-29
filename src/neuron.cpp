#include "../include/neuron.hpp"

template <class T>
Neuron<T>::Neuron(const Neuron& other) {
  weights_ = other.weights_;
  bias_ = other.bias_;
  activation_ = other.activation_;
}

template <typename T>
Neuron<T>::Neuron(const size_t nin, const size_t nout,
                  const UnaryOp& activation)
    : activation_(activation) {
  for (size_t i = 0; i < nin; i++) {
    weights_.emplace_back(
        Value<T>(generate_weight<T>(activation, nin, nout)));
    bias_ = Value<T>(1e-5);
  }
}

template <typename T>
Neuron<T>::Neuron(Neuron&& other) noexcept {
  weights_ = std::move(other.weights_);
  bias_ = std::move(other.bias_);
  activation_ = other.activation_;
}

template <typename T>
Neuron<T>& Neuron<T>::operator=(const Neuron& other) {
  if (this != &other) {
    weights_ = other.weights_;
    bias_ = other.bias_;
    activation_ = other.activation_;
  }
  return *this;
}

template <typename T>
Neuron<T>& Neuron<T>::operator=(Neuron&& other) noexcept {
  if (this != &other) {
    weights_ = std::move(other.weights_);
    bias_ = std::move(other.bias_);
    activation_ = other.activation_;
  }
  return *this;
}

template <typename T>
ParamVector<T> Neuron<T>::get_parameters() const  {
  ParamVector<T> rvec;
  rvec.reserve(weights_.size() + 1);
  for (const auto& w : weights_) {
    rvec.emplace_back(std::make_shared<Value<T>>(w));
  }
  rvec.emplace_back(std::make_shared<Value<T>>(bias_));
  return rvec;
}

template <typename T>
Value<T> Neuron<T>::operator()(const std::vector<Value<T>>& input) const {
  if (input.size() != weights_.size()) {
    throw std::invalid_argument(
        "Vector sizes must be of equal length for dot product calculation.");
  }
  Value<T> rval = bias_;
  for (size_t i = 0; i < input.size(); i++) {
    rval += input[i] * weights_[i];
  }
  if (activation_ == UnaryOp::relu)
    return relu(rval);
  return rval;
}

template <typename T>
Value<T> Neuron<T>::operator()(const std::vector<T>& input) const {
  auto rval = bias_;
  for (size_t i = 0; i < input.size(); i++) {
    rval += input[i] * weights_[i];
  }
  return rval;
}

template
class Neuron<double>;

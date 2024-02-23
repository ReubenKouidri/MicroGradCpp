#ifndef LAYER_HPP
#define LAYER_HPP

#include <vector>
#include <ranges>
#include "module.hpp"
#include "neuron.hpp"

template <typename T>
class Layer final : public Module<T> {
  std::vector<Neuron<T>> neurons_;
  UnaryOp activation_;
  std::size_t num_params_;

 public:
  Layer(std::size_t, std::size_t, const UnaryOp&);
  Layer(const Layer&);
  Layer(Layer&&) noexcept;
  Layer& operator=(const Layer&);
  Layer& operator=(Layer&&) noexcept;
  Output<T> operator()(const std::vector<Value<T>>&) const;
  Output<T> operator()(const std::vector<T>&) const;
  [[nodiscard]] ParamVector<T> get_parameters() const override;
  [[nodiscard]] constexpr std::size_t num_params() const noexcept { return num_params_; }
  [[nodiscard]] std::vector<T> predict(const std::vector<T>&) const;
};

template <typename T>
Layer<T>::Layer(const std::size_t nin,
                const std::size_t nout,
                const UnaryOp& activation)
  : activation_(activation),
    num_params_(nout * (nin + 1)) {
  neurons_.reserve(nout);
  for (std::size_t i = 0; i < nout; i++) {
    neurons_.emplace_back(Neuron<T>(nin,
                                    nout,
                                    activation));
  }
}

template <typename T>
Layer<T>::Layer(const Layer& other)
  : neurons_(other.neurons_),
    activation_(other.activation_),
    num_params_(other.num_params_) {}

template <typename T>
Layer<T>::Layer(Layer&& other) noexcept
  : neurons_(std::move(other.neurons_)),
    activation_(std::move(other.activation_)),
    num_params_(other.num_params_) {}

template <typename T>
Layer<T>& Layer<T>::operator=(const Layer& other) {
  if (this != &other) {
    neurons_ = other.neurons_;
    activation_ = other.activation_;
    num_params_ = other.num_params_;
  }
  return *this;
}

template <typename T>
Layer<T>& Layer<T>::operator=(Layer&& other) noexcept {
  if (this != &other) {
    neurons_ = std::move(other.neurons_);
    activation_ = std::move(other.activation_);
    num_params_ = std::move(other.num_params_);
  }
  return *this;
}

template <typename T>
ParamVector<T> Layer<T>::get_parameters() const {
  ParamVector<T> params;
  params.reserve(num_params_);
  for (const auto& n : neurons_) {
    auto temp = n.get_parameters();
    params.insert(params.end(), temp.begin(), temp.end());
  }
  return params;
}

template <typename T>
Output<T> Layer<T>::operator()(const std::vector<Value<T>>& inputs) const {
  Output<T> output;
  output.reserve(neurons_.size());
  std::ranges::transform(neurons_, std::back_inserter(output),
                         [&inputs](const auto& n) { return n(inputs); });

  if (activation_ == UnaryOp::softmax)
    softmax(output);
  return output;
}

template <typename T>
Output<T> Layer<T>::operator()(const std::vector<T>& input) const {
  std::vector<Value<T>> new_input;
  new_input.reserve(input.size());
  for (const auto t : input) { new_input.emplace_back(Value(t)); }
  return operator()(new_input);
}

template <typename T>
[[nodiscard]] std::vector<T> Layer<T>::predict(const std::vector<T>& input) const {
  std::vector<T> out;
  out.reserve(neurons_.size());
  for (const auto& n : neurons_) { out.emplace_back(n.predict(input)); }
  return out;
}

template
class Layer<double>;

#endif //LAYER_HPP
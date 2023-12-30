#include "../include/model.hpp"

template <typename T>
MLP<T>::MLP(const MLP &other) : layers_(other.layers_) {}

template <typename T>
MLP<T>::MLP(std::vector<Layer<T>> layers) : layers_(std::move(layers)) {}

template <class T>
ParamVector<T> MLP<T>::get_parameters() const {
  ParamVector<T> params;
  for (const auto &l : layers_) {
    auto lparams = l.get_parameters();
    params.insert(params.end(), lparams.begin(), lparams.end());
  }
  return params;
}

template <class T>
void MLP<T>::step(const double learning_rate) const {
  for (const auto &p : get_parameters()) {
    p->step(learning_rate);
  }
}

template <class T>
void MLP<T>::zero_grad() const {
  for (const auto &p : get_parameters()) {
    p->zero_grad();
  }
}

template <class T>
Output<T> MLP<T>::operator()(const std::vector<Value<T>> &inputs) const {
  auto output = inputs;
  for (const auto &l : layers_) {
    output = l(output);
  }
  return output;
}

template <typename T>
Output<T> MLP<T>::operator()(const std::vector<T> &input) const {
  std::vector<Value<T>> new_input;
  new_input.reserve(input.size());
  for (const auto &val : input) {
    new_input.emplace_back(Value(static_cast<T>(val)));
  }
  return operator()(new_input);
}

template
class MLP<double>;

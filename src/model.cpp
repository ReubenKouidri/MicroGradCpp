#include "../include/model.hpp"

template <typename T>
MLP<T>::MLP(const MLP &other) : layers_(other.layers_) {}

template <typename T>
MLP<T>::MLP(const std::vector<Layer<T>> &layers) : layers_(layers) {}

template <typename T>
MLP<T>::MLP(std::vector<Layer<T>> &&layers) noexcept {
  layers_ = std::move(layers);
}

template <class T>
[[nodiscard]] ParamVector<T> MLP<T>::get_parameters() const {
  ParamVector<T> params;
  size_t total_params = 0;
  for (const auto &l : layers_) total_params += l.num_params();
  params.reserve(total_params);
  for (const auto &l : layers_) {
    auto lparams = l.get_parameters();
    params.insert(params.end(), lparams.begin(), lparams.end());
  }
  return params;
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
  for (const auto &val : input)
    new_input.emplace_back(Value(static_cast<T>(val), false));
  return operator()(new_input);
}

template <typename T>
uint8_t MLP<T>::predict(const std::vector<T> &input) const {
  std::vector<T> out = input;
  for (const auto &layer : layers_) {
    out = layer.predict(out);
  }
  auto max_iter = std::max_element(out.begin(), out.end());
  if (max_iter != out.end()) {
    return static_cast<uint8_t>(std::distance(out.begin(), max_iter));
  } else {
    std::cout << "Vector is empty" << std::endl;
    return 255;
  }
}

template
class MLP<double>;

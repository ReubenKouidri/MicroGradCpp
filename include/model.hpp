#ifndef MODEL_HPP
#define MODEL_HPP

#include "module.hpp"
#include "layer.hpp"
#include "neuron.hpp"

template <typename T>
class MLP final : public Module<T> {
  std::vector<Layer<T>> layers_;

 public:
  MLP(const MLP&);
  MLP& operator=(const MLP&) noexcept = delete;
  MLP(MLP&&) noexcept = delete;
  MLP& operator=(MLP&&) noexcept = delete;
  explicit MLP(const std::vector<Layer<T>>&);
  explicit MLP(std::vector<Layer<T>>&&) noexcept;
  ParamVector<T> get_parameters() const override;
  void zero_grad() const;
  Output<T> operator()(const std::vector<Value<T>>&) const;
  Output<T> operator()(const std::vector<T>&) const;
  uint8_t predict(const std::vector<T>&) const;
};


template <typename T>
MLP<T>::MLP(const MLP &other) : layers_(other.layers_) {}

template <typename T>
MLP<T>::MLP(const std::vector<Layer<T>> &layers) : layers_(layers) {}

template <typename T>
MLP<T>::MLP(std::vector<Layer<T>> &&layers) noexcept {
  layers_ = std::move(layers);
}

template <typename T>
[[nodiscard]] ParamVector<T> MLP<T>::get_parameters() const {
  ParamVector<T> params;
  std::size_t total_params = 0;
  for (const auto &l : layers_) total_params += l.num_params();
  params.reserve(total_params);
  for (const auto &l : layers_) {
    auto lparams = l.get_parameters();
    params.insert(params.end(), lparams.begin(), lparams.end());
  }
  return params;
}

template <typename T>
void MLP<T>::zero_grad() const {
  for (const auto &p : get_parameters()) {
    p->zero_grad();
  }
}

template <typename T>
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


#endif //MODEL_HPP
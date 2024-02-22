#ifndef MODEL_HPP
#define MODEL_HPP

#include "module.hpp"
#include "layer.hpp"
#include "neuron.hpp"

template <typename T>
class MLP final : public Module<T> {
  std::vector<Layer<T>> layers_;

public:
  MLP(const MLP &other);
  MLP &operator=(const MLP &other) noexcept = delete;
  MLP(MLP &&other) noexcept = delete;
  MLP &operator=(MLP &&other) noexcept = delete;
  explicit MLP(const std::vector<Layer<T>> &layers);
  explicit MLP(std::vector<Layer<T>> &&layers) noexcept;
  ParamVector<T> get_parameters() const override;
  void zero_grad() const;
  Output<T> operator()(const std::vector<Value<T>> &inputs) const;
  Output<T> operator()(const std::vector<T> &input) const;
  uint8_t predict(const std::vector<T> &input) const;
};

#endif //MODEL_HPP
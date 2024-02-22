#ifndef NEURON_HPP
#define NEURON_HPP

#include <vector>
#include "module.hpp"

template <typename T>
class Neuron final : public Module<T> {
  std::vector<Value<T>> weights_;
  Value<T> bias_;
  UnaryOp activation_{UnaryOp::relu};

 public:
  Neuron(std::size_t nin, std::size_t nout, const UnaryOp &activation);
  Neuron(const Neuron &other);
  Neuron(Neuron &&other) noexcept;
  Neuron &operator=(const Neuron &other);
  Neuron &operator=(Neuron &&other) noexcept;
  [[nodiscard]] ParamVector<T> get_parameters() const override;
  Value<T> operator()(const std::vector<Value<T>> &input) const;
  Value<T> operator()(const std::vector<T> &input) const;
  [[nodiscard]] T predict(const std::vector<T> &input) const;
};

#endif //NEURON_HPP

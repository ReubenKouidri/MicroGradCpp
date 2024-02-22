#ifndef LAYER_HPP
#define LAYER_HPP

#include <vector>
#include "module.hpp"
#include "neuron.hpp"

template <typename T>
class Layer final : public Module<T> {
  std::vector<Neuron<T>> neurons_;
  UnaryOp activation_;
  std::size_t num_params_;

public:
  Layer(std::size_t nin, std::size_t nout, const UnaryOp &activation);
  Layer(const Layer &other);
  Layer(Layer &&other) noexcept;
  Layer &operator=(const Layer &other);
  Layer &operator=(Layer &&other) noexcept;
  Output<T> operator()(const std::vector<Value<T>> &inputs) const;
  Output<T> operator()(const std::vector<T> &input) const;
  [[nodiscard]] ParamVector<T> get_parameters() const override;
  [[nodiscard]] constexpr std::size_t num_params() const noexcept { return num_params_; }
  [[nodiscard]] std::vector<T> predict(const std::vector<T> &input) const;
};

#endif //LAYER_HPP
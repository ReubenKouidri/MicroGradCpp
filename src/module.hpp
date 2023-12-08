#ifndef MODULE_HPP
#define MODULE_HPP

#include <random>
#include "value.hpp"

template<class T>
using Output = std::vector<Value<T>>;

template<class T>
using ParamVector = std::vector<std::shared_ptr<Value<T>>>;

// interface
template<class T>
class Module {
public:
  virtual ~Module()= default;
  Module() = default;
  Module(const Module& other) = default;
  Module(Module&& other) noexcept = default;
  Module& operator=(const Module& other) = default;
  Module& operator=(Module&& other) noexcept = default;
  [[nodiscard]] virtual ParamVector<T> get_parameters() const = 0;
  // Output<T> virtual operator()(const std::vector<Value<T>>& input) const = 0;
  // Output<T> virtual operator()(const std::vector<T>& input) const = 0;
};

// Neuron class inheriting from Module
template<class T>
class Neuron final: public Module<T> {
  // dendrites (weights) feed in
  // TODO: use node and edge layout
  std::vector<Value<T>> weights_;
  Value<T> bias_;
  Activation activation_ { Activation::RELU };

public:
  Neuron(const Neuron& other) {
    weights_ = other.weights_;
    bias_ = other.bias_;
    activation_ = other.activation_;
  }
  ~Neuron() override { weights_.clear(); }
  explicit Neuron(const size_t nin) {
    for (size_t i = 0; i < nin; i++) {
      weights_.emplace_back(Value<T>());
    }
    bias_ = Value<T>();
  }
  ParamVector<T> get_parameters() const override {
    ParamVector<T> retvec;
    retvec.reserve(weights_.size() + 1);
    for (const auto& w: weights_) {
      retvec.emplace_back(std::make_shared<Value<T>>(w));
    }
    retvec.emplace_back(std::make_shared<Value<T>>(bias_));
    return retvec;
  }
  Value<T> operator()(const std::vector<Value<T>>& input) const {
    if (input.size() != weights_.size()) {
      throw std::invalid_argument("Vector sizes must be of equal length for dot product calculation.");
    }
    Value<T> rval = bias_;
    for (size_t i = 0; i < input.size(); i++) {
      auto temp = input[i] * weights_[i];
      rval = rval + temp;
    }
    return rval.activation_output(activation_);
  }
  Value<T> operator()(const std::vector<T>& input) const {
    std::vector<Value<T>> new_input;
    for (const auto& i : input) {
      new_input.emplace_back(Value<T>(i));
    }
    return operator()(new_input);
  }
};


template<class T>
class Layer final: public Module<T> {
  size_t nin_;
  size_t nout_;
  std::vector<Neuron<T>> neurons_;
public:
  Layer(const size_t nin, const size_t nout): nin_(nin), nout_(nout) {
    neurons_.reserve(nout);
    for (size_t i = 0; i < nout; i++) {
      neurons_.emplace_back(Neuron<T>(nin));
    }
  }
  ~Layer() override { neurons_.clear(); }

  Output<T> operator()(const std::vector<Value<T>>& inputs) const {
    Output<T> output;
    for (const auto& n: neurons_) {
      output.emplace_back(n(inputs));
    }
    return output;
  }
  Output<T> operator()(const std::vector<T>& inputs) const {
    return operator()(static_cast<Value<T>>(inputs));
  }
  ParamVector<T> get_parameters() const override {
    ParamVector<T> params;
    for (const auto& n: neurons_) {
      auto temp = n.get_parameters();
      params.insert(params.end(), temp.begin(), temp.end());
    }
    return params;
  }
};

template<class T>
class MLP final: public Module<T> {
  std::vector<Layer<T>> layers_;

public:
  MLP(const MLP& other) : Module<T>(other), layers_(other.layers_) {}
  MLP(MLP&& other) noexcept:
    Module<T>(std::move(other)),
    layers_(std::move(other.layers_)) {
  }
  MLP& operator=(MLP other) {
    std::swap(*this, other);
    return *this;
  }
  ~MLP() override { layers_.clear(); }
  explicit MLP(const std::vector<size_t>& sizes) {
    layers_.reserve(sizes.size());
    for (size_t idx = 0; idx < sizes.size() - 1; ++idx) {
      layers_.emplace_back(sizes[idx], sizes[idx + 1]);
    }
  }
  ParamVector<T> get_parameters() const override {
    ParamVector<T> params;
    for (const auto& l: layers_) {
      auto lparams = l.get_parameters();
      params.insert(params.end(), lparams.begin(), lparams.end());
    }
    return params;
  }
  void step(const double learning_rate) const {
    for (const auto& p : get_parameters()) {
      p->step(learning_rate);
    }
  }
  void zero_grad() const {
    for (const auto& p: get_parameters()) {
      p->zero_grad();
    }
  }

  Output<T> operator()(const std::vector<Value<T>>& inputs) const {
    Output<T> output = inputs;
    for (auto& l : layers_) {
      output = l(output);
    }
    return output;
  }
  Output<T> operator()(const std::vector<T>& inputs) const {
    std::vector<Value<T>> new_inputs;
    new_inputs.reserve(inputs.size());
    for (const auto& i : inputs) {
      auto temp = Value<T>(i);
    }
    return operator()(static_cast<std::vector<Value<T>>>(inputs));
  }

};

#endif //MODULE_HPP

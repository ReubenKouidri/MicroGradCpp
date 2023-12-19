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
  virtual ~Module() = default;
  Module() = default;
  Module(const Module& other) = default;
  Module(Module&& other) noexcept = default;
  Module& operator=(const Module& other) = default;
  Module& operator=(Module&& other) noexcept = default;
  [[nodiscard]] virtual ParamVector<T> get_parameters() const = 0;
};

// Neuron class inheriting from Module
template<class T>
class Neuron final: public Module<T> {
  std::vector<Value<T>> weights_;
  Value<T> bias_;
  UnaryOp activation_ { UnaryOp:: relu };

public:
  Neuron(const Neuron& other) {
    weights_ = other.weights_;
    bias_ = other.bias_;
    activation_ = other.activation_;
  }
  ~Neuron() override { weights_.clear(); }
  explicit Neuron(const size_t nin, const size_t nout, const UnaryOp& activation)
    : activation_(activation) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<T> dist;

    switch (activation_) {
      case UnaryOp::relu:
        /* He initialization */
          dist = std::normal_distribution<T>(0, std::sqrt(2.0 / static_cast<double>(nin)));
        break;
      case UnaryOp::softmax:
        /* Xavier init */
        dist = std::normal_distribution<T>(0, std::sqrt(2.0 / static_cast<double>(nin + nout)));
        break;
      default:
        std::cout << "Need to implement init method in Neuron constructor\n";
        break;
    }

    for (size_t i = 0; i < nin; i++) {
      weights_.emplace_back(Value<T>(dist(gen))); // Use He initialization for weights
    }

    bias_ = Value<T>(1e-5);
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

  Value<T> operator()(const std::vector<T>& input) const {
    Value<T> rval = bias_;
    for (size_t i = 0; i < input.size(); i++) {
      rval += input[i] * weights_[i];
    }
    return rval;
  }
};


template<class T>
class Layer final: public Module<T> {
  std::vector<Neuron<T>> neurons_;
  UnaryOp activation_;
public:
  Layer(const size_t nin, const size_t nout, const UnaryOp& activation)
    : activation_(activation) {
    neurons_.reserve(nout);
    for (size_t i = 0; i < nout; i++) {
      neurons_.emplace_back(Neuron<T>(nin, nout, activation));
    }
  }
  ~Layer() override { neurons_.clear(); }

  Output<T> operator()(const std::vector<Value<T>>& inputs) const {
    Output<T> output;
    for (const auto& n: neurons_) {
      output.emplace_back(n(inputs));
    }
    if (activation_ == UnaryOp::softmax) {
      // for numerical stability, subtract max value
      // this transformation leaves the softmax output invariant
      auto max_val = *std::max_element(output.begin(), output.end(),
                                              [&](const Value<T>& a, const Value<T>& b) {
                                                return a < b;});

      for (auto& o : output)
        o -= max_val;

      auto sum = Value(0.0);
      for (auto& o: output) {
        o = ops::exp(o);
        sum += o;
      }
      for (auto& o: output)
        o /= sum;
    }
    return output;
  }

  Output<T> operator()(const std::vector<T>& input) const {
    std::vector<Value<T>> new_input;
    new_input.reserve(input.size());
    for (const auto t : input) {
      new_input.emplace_back(Value(t));
    }
    return operator()(new_input);
  }
  ParamVector<T> get_parameters() const override {
    ParamVector<T> params;
    for (const auto& n: neurons_) {
      auto temp = n.get_parameters();
      params.insert(params.end(), temp.begin(), temp.end());
    }
    return params;
  }
  [[nodiscard]] size_t get_size() const { return neurons_.size(); }
  std::vector<Neuron<T>>& get_neurons() { return neurons_; }
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

  explicit MLP(std::vector<Layer<T>> layers) : layers_(std::move(layers)) {}

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

  template<class C>
  Output<T> operator()(const std::vector<C>& input) const {
    std::vector<Value<T>> new_input;
    new_input.reserve(input.size());
    for (const auto c : input) {
      new_input.emplace_back(Value(static_cast<T>(c)));
    }
    return operator()(new_input);
  }
};


#endif //MODULE_HPP


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
public:
  static auto rng() {
    static std::random_device rand;
    static std::mt19937 gen(rand());
    static std::uniform_real_distribution<double> dist(0, 1);
    return dist(gen);
  }

private:
  std::vector<Value<T>> weights_;
  Value<T> bias_;
  size_t nin_;
  Activation activation_ { Activation::RELU };

public:
  Neuron(const Neuron& other) {
    nin_ = other.nin_;
    weights_ = other.weights_;
    bias_ = other.bias_;
    activation_ = other.activation_;
  }
  ~Neuron() override { weights_.clear(); }
  explicit Neuron(const size_t nin): nin_(nin) {
    for (size_t i = 0; i < nin; i++) {
      weights_.push_back(Value( rng() ));
    }
    bias_ = Value<T>(rng());
  }
  ParamVector<T> get_parameters() const override {
    ParamVector<T> retvec;
    retvec.reserve(weights_.size() + 1);
    for (const auto& w: weights_) {
      retvec.push_back(std::make_shared<Value<T>>(w));
    }
    retvec.push_back(std::make_shared<Value<T>>(bias_));
    return retvec;
  }
  Value<T> operator()(const std::vector<Value<T>>& input) const {
    if (input.size() != weights_.size()) {
      throw std::invalid_argument("Vector sizes must be of equal length for dot product calculation.");
    }
    Value<T> rval = bias_;
    for (size_t i = 0; i < input.size(); i++) {
      rval += input[i] * weights_[i];
    }
    return rval.activation_output(activation_);
  }
  Value<T> operator()(const std::vector<T>& input) const {
    std::vector<Value<T>> new_input;
    for (const auto& i : input) {
      new_input.push_back(Value<T>(i));
    }
    return operator()(new_input);
  }
};


template<class T>
class Layer final: public Module<T> {
  size_t nin_;
  size_t nout_;
  std::vector<Value<T>> neurons_;
public:
  Layer(const size_t nin, const size_t nout): nin_(nin), nout_(nout) {
    neurons_.reserve(nout);
    for (size_t i = 0; i < nout; i++) {
      neurons_.push_back(Neuron<Value<T>>(nin));
    }
  }
  ~Layer() override { neurons_.clear(); }

  Output<T> operator()(const std::vector<Value<T>>& inputs) const {
    Output<T> output;
    for (const auto& n: neurons_) {
      output.push_back(n(inputs));
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
  MLP(const MLP&) = delete;
  MLP(MLP&&) = delete;
  ~MLP() override { layers_.clear(); }
  explicit MLP(const size_t nin, const std::vector<size_t>& layer_sizes) {
    // nin = numbuer of inputs to first layer of size layer_sizes[0]
    // layer_sizes = number of neurons in each layer
    // layer_sizes[-1] = num classes (output layer)
    auto sizes = layer_sizes;
    sizes.insert(sizes.begin(), nin);
    layers_.reserve(sizes.size());
    for (size_t idx = 0; idx < sizes.size() - 1; idx++) {
      layers_.push_back(Layer<T>(idx, idx + 1));
    }
  }
  ParamVector<T> get_parameters() const override {
    ParamVector<T> params;
    for (const auto& l: layers_) {
      auto lparams = layers_.get_parameters();
      params.insert(params.end(), lparams.begin(), lparams.end());
    }
    return params;
  }
  void backward(const double learning_rate) {
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
    for (const auto& l : layers_) {
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

template<class T>
class MSELoss {
  MLP<T> network_;
  std::vector<Value<T>> targets_;
public:
  explicit MSELoss(const MLP<T>& network, const std::vector<T>& targets): network_(network) {
    targets_.reserve(targets.size());
    for (const auto& t : targets) {
      targets_.push_back(Value<T>(t));
    }
  }
  Value<T> compute_loss(const std::vector<Value<T>>& input) {
    auto output = network_(input);
    Value<T> loss;
    for (size_t i = 0; i < input.size(); i++) {
      loss += pow(input[i] - targets_[i], 2);
    }
    loss /= input.size();
    return loss;
  }
  Value<T> compute_loss(const std::vector<T>& input) {
    std::vector<Value<T>> new_inputs;
    for (const auto& i : input) {
      auto temp = Value<T>(input);
      new_inputs.push_back(temp);
    }
    return compute_loss(new_inputs);
  }
};

#endif //MODULE_HPP

#ifndef LOSSES_HPP
#define LOSSES_HPP

#include "value.hpp"
#include "module.hpp"

// Base class to inherit fromx
template<class T, class Target_Tp>
class Loss {
public:
  using input_type          = std::vector<T>;
  using target_type         = Target_Tp; /* e.g. uint8_t*/
  using batched_input_type  = std::vector<input_type>;
  using batched_target_type = std::vector<target_type>;

protected:
  MLP<T> network_;
  double learning_rate_;
  Value<T> value_ {static_cast<T>(0)};
  static constexpr T eps_ = 1e-7;
  static constexpr T alpha_ = 10;

public:
  explicit Loss(const MLP<T>& network, const double learning_rate)
    : network_(network), learning_rate_(learning_rate) {}
  virtual ~Loss() = default;
  Loss(const Loss& other) = delete;
  Loss(Loss&& other) = delete;
  Loss& operator=(const Loss& other) = delete;
  Loss& operator=(Loss&& other) = delete;

  virtual void compute_loss(const input_type& input,
                            const target_type& target) = 0;
  virtual void compute_loss(const batched_input_type& batched_input,
                            const batched_target_type& batched_target) {
    for (size_t i = 0; i < batched_input.size(); ++i) {
      compute_loss(batched_input[i], batched_target[i]);
    }
    value_ /= static_cast<T>(batched_input.size());
  }

  virtual void zero() { value_ = Value(static_cast<T>(0)); }
  virtual void clamp(Output<T>& output) {
    for (auto& val : output) {
      val.get_data() = std::clamp(val.get_data(),
                                  this->eps_, 1 - this->eps_);
    }
  }
  virtual T get() { return value_.get_data(); }
  virtual void backward() { value_.backward(); }

};

/*============================================================================*/
template<class T>
class SparseCCELoss final : public Loss<T, uint8_t> {
public:
  using Loss =   Loss<T, uint8_t>;
  using typename Loss::input_type;
  using typename Loss::target_type;
  using typename Loss::batched_input_type;
  using typename Loss::batched_target_type;

  using Loss::Loss;  // Inherit constructor
  using Loss::compute_loss;

  // Implement compute_loss for single input
  void compute_loss(const input_type& input,
                    const target_type& target) override {
    auto outputs = this->network_(input);
    this->clamp(outputs);
    this->value_ -= log(outputs[target]);
  }
};

/*============================================================================*/
/* target ohe, e.g. {0,...,1, 0} */
template<class T>
class CCELoss final: public Loss<T, std::vector<uint8_t>> {
public:
  using Loss =    Loss<T, std::vector<uint8_t>>;
  using typename  Loss::input_type;
  using typename  Loss::target_type;
  using typename  Loss::batched_input_type;
  using typename  Loss::batched_target_type;

  using Loss::Loss;
  using Loss::compute_loss;

  size_t get_index(const target_type& target) {
    auto it = std::find(target.begin(),
                                                   target.end(), 1);
    if (it != target.end()) {
      return std::distance(target.begin(), it);
    }
    throw std::runtime_error("Target is not in OHE format. Consider using "\
                             "sparse CCE instead\n");
  }

  void compute_loss(const input_type& input,
                    const target_type& target) override {
    auto output = this->network_(input);
    this->clamp(output);
    auto index = get_index(target);
    this->value_ += -log(output[index]);
  }
};

/*============================================================================*/
/* specialisation for single input, sparse target */

template<class T>
class MSELoss final: public Loss<T, uint8_t> {
public:
  using Loss =   Loss<T, uint8_t>;
  using typename Loss::input_type;
  using typename Loss::target_type;
  using typename Loss::batched_input_type;
  using typename Loss::batched_target_type;

  using Loss::Loss;
  using Loss::compute_loss;

  void compute_loss(const input_type& input,
                    const target_type& target) override {
    auto output = this->network_(input);
    this->clamp(output);

    for (size_t i = 0; i < output.size(); i++) {
      if (i == static_cast<size_t>(target))
        this->value_ += ops::pow(output[i] - 1.0, static_cast<T>(2));
      else
        this->value_ += ops::pow(output[i], static_cast<T>(2));
    }
    this->value_ /= output.size();
  }
};

#endif //LOSSES_HPP
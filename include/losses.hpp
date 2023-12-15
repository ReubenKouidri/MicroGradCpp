#ifndef LOSSES_HPP
#define LOSSES_HPP

#include "value.hpp"
#include "module.hpp"

// Base class to inherit fromx
template<class T, class Target_Tp>
class Loss {
public:
  typedef std::vector<T>              input_type;
  typedef Target_Tp /* e.g. uint8_t*/ target_type;
  typedef std::vector<input_type>     batched_input_type;
  typedef std::vector<target_type>    batched_target_type;

protected:
  MLP<T> network_;
  double learning_rate_;

public:
  explicit Loss(const MLP<T>& network, const double learning_rate)
    : network_(network), learning_rate_(learning_rate) {}
  virtual ~Loss() = default;
  Loss(const Loss& other) = delete;
  Loss(Loss&& other) = delete;
  Loss& operator=(const Loss& other) = delete;
  Loss& operator=(Loss&& other) = delete;

  virtual Value<T> compute_loss(const input_type& input, const target_type& target) = 0;
  virtual Value<T> compute_loss(const batched_input_type& batched_input, const batched_target_type& batched_target) {
    Value<T> total_loss = Value<T>(0.0);
    for (size_t i = 0; i < batched_input.size(); ++i) {
      total_loss += compute_loss(batched_input[i], batched_target[i]);
    }
    return total_loss / static_cast<T>(batched_input.size());
  }

  virtual void step() const { network_.step(learning_rate_); }
  virtual void zero_grad() const { network_.zero_grad(); }
};

/*==================================================================================================================*/
template<class T>
class SparseCCELoss final : public Loss<T, uint8_t> {
public:
  typedef Loss<T, uint8_t> Loss;
  using typename           Loss::input_type;
  using typename           Loss::target_type;
  using typename           Loss::batched_input_type;
  using typename           Loss::batched_target_type;

  using Loss::Loss;  // Inherit constructor

  // Implement compute_loss for single input
  Value<T> compute_loss(const input_type& input, const target_type& target) override {
    auto outputs = this->network_(input); // network output is a softmax vector
    Value<T> loss = -vlog(outputs[target]);
    return loss;
  }

  // Use the base class implementation for batched input
  using Loss::compute_loss;
};

/*==================================================================================================================*/

/* specialisation for single input vector where target is ohe, e.g. {0, 1, 0, ..., 0} */
template<class T>
class CCELoss final: public Loss<T, std::vector<uint8_t>> {
public:
  typedef Loss<T, std::vector<uint8_t>> Loss;
  using typename                        Loss::input_type;
  using typename                        Loss::target_type;
  using typename                        Loss::batched_input_type;
  using typename                        Loss::batched_target_type;

  using Loss::Loss;  // Inherit constructor

  Value<T> compute_loss(const input_type& input,
                        const target_type& target) override {
    auto output = this->network_(input); // network output is a softmax vector
    Value<T> loss = Value<T>(0.0);
    for (size_t i = 0; i < output.size(); ++i) {
      loss += -vlog(output[i]) * Value<T>(target[i]);
    }
    return loss;
  }
  using Loss::compute_loss;
};

/*==================================================================================================================*/
/* specialisation for single input, sparse target */

template<class T>
class MSELoss final: public Loss<T, uint8_t> {
public:
  typedef Loss<T, uint8_t> Loss;
  using typename           Loss::input_type;
  using typename           Loss::target_type;
  using typename           Loss::batched_input_type;
  using typename           Loss::batched_target_type;

  using Loss::Loss;  // Inherit constructor

  Value<T> compute_loss(const input_type& input,
                        const target_type& target) override {
    auto output = this->network_(input);
    auto loss = Value<T>(0.0);
    for (size_t i = 0; i < output.size(); i++) {
      if (i == static_cast<size_t>(target))
        loss += pow(output[i] - 1.0, static_cast<T>(2));
      else
        loss += pow(output[i], static_cast<T>(2));
    }
    loss /= output.size();
    return loss;
  }
  using Loss::compute_loss;
};

#endif //LOSSES_HPP
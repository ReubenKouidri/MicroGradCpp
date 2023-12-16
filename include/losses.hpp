
#ifndef NEW_LOSSES_HPP
#define NEW_LOSSES_HPP

#include "value.hpp"
#include "module.hpp"

// Base class to inherit from
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
  Value<T> loss_obj_;

public:
  explicit Loss(const MLP<T>& network, const double learning_rate)
    : network_(network), learning_rate_(learning_rate) {}
  virtual ~Loss() = default;
  Loss(const Loss& other) = delete;
  Loss(Loss&& other) = delete;
  Loss& operator=(const Loss& other) = delete;
  Loss& operator=(Loss&& other) = delete;

  virtual Value<T>& compute_loss(const input_type& input, const target_type& target) = 0;
  virtual Value<T>& compute_loss(const batched_input_type& batched_input, const batched_target_type& batched_target) {
    for (size_t i = 0; i < batched_input.size(); ++i) {
      loss_obj_ += compute_loss(batched_input[i], batched_target[i]);
    }
    loss_obj_ /= static_cast<T>(batched_input.size());
    return loss_obj_;
  }

  virtual void step() const { network_.step(learning_rate_); }
  virtual void zero_grad() const { network_.zero_grad(); }
  virtual void zero_loss() {
    loss_obj_.get_data() = 0.0;
    loss_obj_.get_grad() = 0.0;
  }
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
  Value<T>& compute_loss(const input_type& input, const target_type& target) override {
    auto outputs = this->network_(input); // network output is a softmax vector
    this->loss_obj_ += -vlog(outputs[target]);
    return this->loss_obj_;
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

  Value<T>& compute_loss(const input_type& input,
                        const target_type& target) override {
    auto output = this->network_(input); // network output is a softmax vector
    for (size_t i = 0; i < output.size(); ++i) {
      this->loss_obj_ += -vlog(output[i]) * Value<T>(target[i]);
    }
    return this->loss_obj_;
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

  Value<T>& compute_loss(const input_type& input,
                        const target_type& target) override {
    auto output = this->network_(input);
    for (size_t i = 0; i < output.size(); i++) {
      if (i == static_cast<size_t>(target))
        this->loss_obj_ += pow(output[i] - 1.0, static_cast<T>(2));
      else
        this->loss_obj_ += pow(output[i], static_cast<T>(2));
    }
    this->loss_obj_ /= output.size();
    return this->loss_obj_;
  }
  using Loss::compute_loss;
};

#endif //NEW_LOSSES_HPP
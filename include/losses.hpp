#ifndef LOSSES_HPP
#define LOSSES_HPP

#include "value.hpp"
#include "module.hpp"

// CRTP
template <class Derived, typename T, class Target_Tp>
class Loss {
 public:
  using input_type = std::vector<T>;
  using target_type = Target_Tp; /* e.g. uint8_t*/
  using batched_input_type = std::vector<input_type>;
  using batched_target_type = std::vector<target_type>;

 protected:
  MLP<T> *model_;
  Value<T> value_{static_cast<T>(0)};
  static constexpr T eps_ = 1e-7;

 public:
  constexpr explicit Loss(MLP<T> *model = nullptr)
      : model_(model) {}
  virtual ~Loss() = default;
  Loss(const Loss &other) = delete;
  Loss(Loss &&other) = delete;
  Loss &operator=(const Loss &other) = delete;
  Loss &operator=(Loss &&other) = delete;

  constexpr void compute_loss(const input_type &input,
                              const target_type &target) {
    static_cast<Derived *>(this)->compute_loss_impl(input, target);
  }
  constexpr void compute_loss(const batched_input_type &batched_input,
                              const batched_target_type &batched_target) {
    for (size_t i = 0; i < batched_input.size(); ++i) {
      compute_loss(batched_input[i], batched_target[i]);
    }
    value_ /= static_cast<T>(batched_input.size());
  }

  constexpr void zero() { value_ = Value(static_cast<T>(0)); }
  virtual void clamp(Output<T> &output) {
    for (auto &val : output) {
      val.get_data() = std::clamp(val.get_data(),
                                  this->eps_, 1 - this->eps_);
    }
  }
  constexpr T get() const { return value_.get_data(); }
  constexpr void backward() { value_.backward(); }
};

/*============================================================================*/
template <class T>
class SparseCCELoss final : public Loss<SparseCCELoss<T>, T, uint8_t> {
  friend class Loss<SparseCCELoss<T>, T, uint8_t>;
 public:
  using Loss = Loss<SparseCCELoss<T>, T, uint8_t>;
  using Loss::Loss;
  using Loss::compute_loss;

  // Implement compute_loss for single input
  constexpr void compute_loss_impl(const Loss::input_type &input,
                                   const Loss::target_type &target) {
    auto outputs = this->model_->operator()(input);
    this->clamp(outputs);
    this->value_ -= log(outputs[target]);
  }
};

/*============================================================================*/
/* target ohe, e.g. {0,...,1, 0} */
template <class T>
class CCELoss final : public Loss<CCELoss<T>, T, std::vector<uint8_t>> {
  friend class Loss<CCELoss<T>, T, std::vector<uint8_t>>;
 public:
  using Loss = Loss<CCELoss<T>, T, std::vector<uint8_t>>;
  using Loss::Loss;
  using Loss::compute_loss;

  size_t get_index(const Loss::target_type &target) const {
    auto it = std::find(target.begin(), target.end(), 1);
    if (it!=target.end()) {
      return std::distance(target.begin(), it);
    }
    throw std::runtime_error("Target is not in OHE format. Consider using "\
                             "sparse CCE instead\n");
  }

  constexpr void compute_loss_impl(const Loss::input_type &input,
                                   const Loss::target_type &target) {
    auto output = this->model_->operator()(input);
    this->clamp(output);
    auto index = get_index(target);
    this->value_ += -log(output[index]);
  }
};

/*============================================================================*/
/* specialisation for single input, sparse target */

template <class T>
class MSELoss final : public Loss<MSELoss<T>, T, uint8_t> {
  friend class Loss<MSELoss<T>, T, uint8_t>;
 public:
  using Loss = Loss<MSELoss<T>, T, uint8_t>;
  using Loss::Loss;
  using Loss::compute_loss;

  constexpr void compute_loss_impl(const Loss::input_type &input,
                                   const Loss::target_type &target) {
    auto output = this->model_->operator()(input);
    this->clamp(output);

    for (size_t i = 0; i < output.size(); i++) {
      if (i==static_cast<size_t>(target))
        this->value_ += ops::pow(output[i] - 1.0, static_cast<T>(2));
      else
        this->value_ += ops::pow(output[i], static_cast<T>(2));
    }
    this->value_ /= output.size();
  }
};

#endif //LOSSES_HPP
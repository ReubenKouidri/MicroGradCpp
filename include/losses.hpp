#ifndef LOSSES_HPP
#define LOSSES_HPP

#include <chrono>
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
  const std::shared_ptr<const MLP<T>> mptr_;
  Value<T> value_{static_cast<T>(0)};
  static constexpr T eps_{1e-7};

 public:
  constexpr explicit Loss(const std::shared_ptr<const MLP<T>> mptr)
      : mptr_(mptr) {}
  virtual ~Loss() = default;
  Loss(const Loss &other) = delete;
  Loss(Loss &&other) = delete;
  Loss &operator=(const Loss &other) = delete;
  Loss &operator=(Loss &&other) = delete;

  constexpr void compute_loss(const input_type &input,
                              const target_type &target) {
    static_cast<Derived *>(this)->compute_loss_impl(input, target);
  }
  void compute_loss(const batched_input_type &batched_input,
                    const batched_target_type &batched_target) {
    for (size_t i = 0; i < batched_input.size(); ++i) {
      compute_loss(batched_input[i], batched_target[i]);
    }
    value_ /= static_cast<T>(batched_input.size());
  }

  constexpr void zero() noexcept { value_ = Value(static_cast<T>(0)); }
  virtual void clamp(Output<T> &output) {
    for (auto &val : output) {
      val.get_data() = std::clamp(val.get_data(),
                                  this->eps_, 1 - this->eps_);
    }
  }
  constexpr T get() const noexcept { return value_.get_data(); }
  void backward() noexcept { value_.backward(); }
};

/*============================================================================*/
template <typename T>
class SparseCCELoss final : public Loss<SparseCCELoss<T>, T, uint8_t> {
  friend class Loss<SparseCCELoss<T>, T, uint8_t>;
 public:
  using Loss = Loss<SparseCCELoss<T>, T, uint8_t>;
  using Loss::Loss;
  using Loss::compute_loss;

  // Implement compute_loss for single input
  void compute_loss_impl(const Loss::input_type &input,
                         const Loss::target_type &target) {
    auto outputs = this->mptr_->operator()(input);
    this->clamp(outputs);
    this->value_ -= log(outputs[target]);
  }
};

/*============================================================================*/
/* target ohe, e.g. {0,...,1, 0} */
template <typename T>
class CCELoss final : public Loss<CCELoss<T>, T, std::vector<uint8_t>> {
  friend class Loss<CCELoss<T>, T, std::vector<uint8_t>>;
 public:
  using Loss = Loss<CCELoss<T>, T, std::vector<uint8_t>>;
  using Loss::Loss;
  using Loss::compute_loss;

  static size_t get_index(const std::vector<uint8_t> &target) {
    auto it = std::find(target.begin(), target.end(), 1);
    if (it!=target.end()) {
      return std::distance(target.begin(), it);
    }
    throw std::runtime_error("Target is not in OHE format. Consider using "\
                             "sparse CCE instead\n");
  }

  void compute_loss_impl(const Loss::input_type &input,
                         const Loss::target_type &target) {
    auto output = this->mptr_->operator()(input);
    this->clamp(output);
    const auto index = get_index(target);
    this->value_ -= log(output[index]);
  }
};

/*============================================================================*/
template <typename T>
class MSELoss final : public Loss<MSELoss<T>, T, uint8_t> {
  friend class Loss<MSELoss<T>, T, uint8_t>;
 public:
  using Loss = Loss<MSELoss<T>, T, uint8_t>;
  using Loss::Loss;
  using Loss::compute_loss;

  void compute_loss_impl(const Loss::input_type &input,
                         const Loss::target_type &target) {
    auto output = this->mptr_->operator()(input);
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
#ifndef OPTIMISER_HPP
#define OPTIMISER_HPP

#include <cmath>
#include <iostream>
#include <memory>
#include <vector>
#include "module.hpp"
#include "value.hpp"

// CRTP base class
template <class Derived, typename T>
class Optimiser {
 protected:
  const std::shared_ptr<const MLP<T>> mptr_;
  const double step_size_;
  std::size_t t_{};
  const double clip_val_;

 public:
  constexpr explicit Optimiser(const std::shared_ptr<const MLP<T>> model,
                               const double step_size,
                               const double clip_val)
      : mptr_(model),
        step_size_(step_size),
        clip_val_(clip_val) {}

  virtual ~Optimiser() = default;

  void step() {
    static_cast<Derived *>(this)->step_impl();
  }
  void zero_grad();
};

template <typename T>
class Adam final : public Optimiser<Adam<T>, T> {
  friend class Optimiser<Adam, T>;  // grant access to base

  const double beta_1_;
  const double beta_2_;
  const double eps_;
  std::vector<double> m_;
  std::vector<double> v_;

 public:
  constexpr explicit Adam(const std::shared_ptr<const MLP<T>> model,
                          const double step_size = 1e-3,
                          const double beta_1 = 0.9, const double beta_2 = 0.999,
                          const double eps = 1e-8, const double clip_val = 1.0)
      : Optimiser<Adam, T>(model, step_size, clip_val),
        beta_1_(beta_1),
        beta_2_(beta_2),
        eps_(eps) {
    const auto size = this->mptr_->get_parameters().size();
    m_ = std::vector<double>(size, 0);
    v_ = std::vector<double>(size, 0);
  }

  void step_impl();
};

#endif //OPTIMISER_HPP

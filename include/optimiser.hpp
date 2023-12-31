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
  MLP<T> *model_;
  double step_size_;
  size_t t_{};
  double clip_val_;

 public:
  constexpr explicit Optimiser(MLP<T> *model, double step_size, double clip_val)
      : model_(model),
        step_size_(step_size),
        clip_val_(clip_val) {}

  virtual ~Optimiser() = default;

  constexpr void step() {
    static_cast<Derived *>(this)->step_impl();
  }

  void zero_grad();
};

template <typename T>
class Adam final : public Optimiser<Adam<T>, T> {
  friend class Optimiser<Adam<T>, T>;  // grant access to base

  double beta_1_;
  double beta_2_;
  double eps_;
  std::vector<double> m_;
  std::vector<double> v_;

 public:
  constexpr explicit Adam(MLP<T> *model, double step_size = 1e-3,
                          double beta_1 = 0.9, double beta_2 = 0.999,
                          double eps = 1e-8, double clip_val = 1.0)
      : Optimiser<Adam<T>, T>(model, step_size, clip_val),
        beta_1_(beta_1),
        beta_2_(beta_2),
        eps_(eps) {
    const auto size = this->model_->get_parameters().size();
    m_ = std::vector<double>(size, 0);
    v_ = std::vector<double>(size, 0);
  }

  void step_impl();
};

#endif //OPTIMISER_HPP

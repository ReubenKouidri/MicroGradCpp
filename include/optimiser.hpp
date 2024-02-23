#ifndef OPTIMISER_HPP
#define OPTIMISER_HPP

#include <cmath>
#include <iostream>
#include <memory>
#include <vector>
#include "module.hpp"
#include "value.hpp"
#include "optimiser.hpp"
#include "model.hpp"

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
                          const double beta_1 = 0.9,
                          const double beta_2 = 0.999,
                          const double eps = 1e-8,
                          const double clip_val = 1.0)
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

template <class Derived, typename T>
void Optimiser<Derived, T>::zero_grad() {
  for (const auto& p : mptr_->get_parameters())p->zero_grad();
}

template <typename T>
void Adam<T>::step_impl() {
  /**
    \name Adam Optimiser
    \details
    m0 ← 0 (Initialize 1st moment vector) \n
    v0 ← 0 (Initialize 2nd moment vector) \n
    while θ_t not converged do \n
        t ← t + 1 \n
        g_t ← ∇f_t(θ_t−1) (Get gradients w.r.t. stochastic objective at timestep t) \n
        m_t ← β1 · m_t−1 + (1 − β1) · g_t (Update biased first moment estimate) \n
        v_t ← β2 · v_t−1 + (1 − β2) · g_t^2 (Update biased second raw moment estimate) \n
        α_t = α · √(1 − β2^t) / (1 − β1^t) \n
        θ_t ← θ_t−1 − α_t * m_t / (√v_t + ε') | ε' := ε√(1-β2^t) \n
    end while \n
    **/

  ++this->t_;

  const auto alpha_t = this->step_size_ *
    std::sqrt(1 - std::pow(beta_2_, this->t_)) /
    (1 - std::pow(beta_1_, this->t_));

  const double eps_p = eps_ * std::sqrt(1 - std::pow(beta_2_, this->t_));

  const auto& params = this->mptr_->get_parameters();
  std::ranges::for_each(params, [this](auto& param) {
    param->get_grad() = std::clamp(
      param->get_grad(), -this->clip_val_, this->clip_val_);
  });

  for (std::size_t i = 0; i < m_.size(); i++) {
    m_[i] = beta_1_ * m_[i] + (1 - beta_1_) * params[i]->get_grad();
    v_[i] = beta_2_ * v_[i] + (1 - beta_2_) *
      std::pow(params[i]->get_grad(), 2);
    params[i]->get_data() -= alpha_t * m_[i] / (std::sqrt(v_[i]) + eps_p);
  }
}

template
class Optimiser<Adam<double>, double>;

template
class Adam<double>;

#endif //OPTIMISER_HPP

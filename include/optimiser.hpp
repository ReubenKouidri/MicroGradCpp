#ifndef OPTIMISER_HPP
#define OPTIMISER_HPP

#include <cmath>
#include <iostream>
#include <memory>
#include <vector>
#include "module.hpp"
#include "value.hpp"

template <typename T>
class Optimiser {
protected:
  std::unique_ptr<MLP<T>> model_;
  double step_size_;
  size_t t_ {};
  double clip_val_; 
public:
  Optimiser(MLP<T> *model, const double step_size, const double clip_val)
    : Optimiser(std::make_unique<MLP<T>>(*model),
                step_size,
                clip_val) {}

  Optimiser(std::unique_ptr<MLP<T>> model,
            const double step_size,
            const double clip_val)
    : model_(std::move(model)),
      step_size_(step_size),
      clip_val_(clip_val) {}

  Optimiser(const Optimiser& other) = delete;
  Optimiser(Optimiser&& other) = delete;
  Optimiser& operator=(const Optimiser& other) = delete;
  Optimiser& operator=(Optimiser&& other) = delete;
  ~Optimiser() = default;

  virtual void step() = 0;
  virtual void zero_grad() {
    for (const auto& p : model_->get_parameters())
      p->zero_grad();
  }
};

template <typename T>
class Adam final : public Optimiser<T> {
  double beta_1_;
  double beta_2_;
  double eps_;
  std::vector<double> m_;
  std::vector<double> v_;
public:
  explicit Adam(MLP<T> *model,
                const double step_size = 1e-3,
                const double beta_1 = 0.9,
                const double beta_2 = 0.999,
                const double eps = 1e-8,
                const double clip_val = 1.0)
      : Optimiser<T>(model, step_size, clip_val),
        beta_1_(beta_1),
        beta_2_(beta_2),
        eps_(eps) {
    const auto size = this->model_->get_parameters().size();
    m_ = std::vector<double>(size, 0);
    v_ = std::vector<double>(size, 0);
  }

  Adam(const Adam& other) = delete;
  Adam(Adam&& other) = delete;
  Adam& operator=(const Adam& other) = delete;
  Adam& operator=(Adam&& other) = delete;
  ~Adam() = default;

  void step() override {
    /**
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
    return θ_t (Resulting parameters)
    **/

    this->t_++;
    const auto& params = this->model_->get_parameters();

    size_t idx = 0;
    for (auto& param : params) {
      T& grad = param->get_grad();
      if (std::abs(grad) > this->clip_val_)
        grad = grad > 0 ? this->clip_val_ : -this->clip_val_;
      idx++;
    }

    for (size_t i = 0; i < m_.size(); i++) {
      m_[i] = beta_1_ * m_[i] + (1 - beta_1_) * params[i]->get_grad();
      v_[i] = beta_2_ * v_[i] + (1 - beta_2_) *
          std::pow(params[i]->get_grad(), 2);
    }

    const auto alpha_t =  this->step_size_ *
                          std::sqrt(1 - std::pow(beta_2_, this->t_)) /
                          (1 - std::pow(beta_1_, this->t_));
    const double eps_p = eps_ * std::sqrt(1 - std::pow(beta_2_, this->t_));

    idx = 0;
    for (auto& param : params) {
      param->get_data() -= alpha_t * m_[idx] / (std::sqrt(v_[idx]) + eps_p);
      ++idx;
    }
  }
};

#endif //OPTIMISER_HPP

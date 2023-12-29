#ifndef OPTIMISER_HPP
#define OPTIMISER_HPP

#include <cmath>
#include <iostream>
#include <memory>
#include <vector>
#include <ranges>
#include "module.hpp"
#include "value.hpp"

template <typename T>
class Optimiser {
protected:
  std::unique_ptr<MLP<T>> model_;
  double step_size_;
  size_t t_{};
  double clip_val_;

public:
  Optimiser(MLP<T>* model, double step_size, double clip_val);

  virtual ~Optimiser() = default;

  virtual void step() = 0;

  virtual void zero_grad();
};

template <typename T>
class Adam final : public Optimiser<T> {
  double beta_1_;
  double beta_2_;
  double eps_;
  std::vector<double> m_;
  std::vector<double> v_;

public:
  explicit Adam(MLP<T>* model, double step_size = 1e-3,
                double beta_1 = 0.9, double beta_2 = 0.999,
                double eps = 1e-8, double clip_val = 1.0);

  ~Adam() override;

  void step() override;
};

#endif //OPTIMISER_HPP

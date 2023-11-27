#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <iostream>
#include <vector>
#include <set>
#include <functional>
#include <ranges>
#include "grad_utils.hpp"


inline std::function do_nothing = []{};

template<typename T>
class BaseTensor {
  template<typename C>
  friend std::ostream& operator<<(std::ostream& os, const BaseTensor<C>& val) {
    os << "Tensor(" << val.get_data() << ", " << val.get_grad() << ")";
    return os;
  }

  T data_ { static_cast<T>(0) };
  T grad_ { static_cast<T>(0) };
  std::set<std::shared_ptr<BaseTensor>> parents_;
  std::function<void()> backward_ { do_nothing };

public:
  BaseTensor(const T& data, const std::set<std::shared_ptr<BaseTensor>>& parents):
    data_{data}, parents_{parents} {}
  explicit BaseTensor(const T& data): data_(data) {}
  ~BaseTensor() = default;
  BaseTensor(const BaseTensor&) = delete;
  BaseTensor(BaseTensor&&) = delete;
  BaseTensor& operator=(const BaseTensor& other) = delete;
  BaseTensor& operator=(BaseTensor&& other) = delete;

  const T& get_data() const { return data_; }
  const T& get_grad() const { return grad_; }
  T& get_data() { return data_; }
  T& get_grad() { return grad_; }
  const std::set<std::shared_ptr<BaseTensor>>& get_parent_ptrs() const { return parents_; }
  void zero_grad() { grad_ = static_cast<T>(0); }
  void zero_grad_all() {
    const auto order = build_topological_order();
    for (auto n = order.rbegin(); n != order.rend(); ++n)
      (*n)->zero_grad();
  }

  void set_backward(const std::function<void()>& func) { backward_ = func; }
  void step(const T& learning_rate) { data_ -= learning_rate * grad_; }

  std::vector<BaseTensor*> build_topological_order() {
    std::vector<BaseTensor*> topological_order;
    std::set<BaseTensor*> visited_nodes;
    // declare lambda
    std::function<void(BaseTensor*, std::set<BaseTensor*>&, std::vector<BaseTensor*>&)> traverse_and_build_order;
    traverse_and_build_order = [&traverse_and_build_order](
      BaseTensor* node, std::set<BaseTensor*>& visited, std::vector<BaseTensor*>& order) {
      if (visited.contains(node))
        return;

      visited.insert(node);
      for (const auto& parent_ptr: node->get_parent_ptrs())
        traverse_and_build_order(parent_ptr.get(), visited, order);
      order.push_back(node);
    };
    traverse_and_build_order(this, visited_nodes, topological_order);
    return topological_order;
  }

  void backward() {
    const auto topological_order = build_topological_order();
    grad_ = static_cast<T>(1); // Set dx/dx=1
    for (const auto& node: std::ranges::reverse_view(topological_order)) {
      if (node != nullptr) node->backward_();
      else { return; }
    }
  }
};

using namespace gradient_ops;

template<typename T>
class Tensor {

  template<typename C> friend std::ostream& operator<<(std::ostream& os, const Tensor<C>& val) {
    os << "Tensor(" << val.get_data() << ", " << val.get_grad() << ")";
    return os;
  }

  // 'right-op' overloads
  template<class C> friend Tensor<C> operator+(C num, const Tensor<C>& val) { return val + num; }
  template<class C> friend Tensor<C> operator-(C num, const Tensor<C>& val) { return val - num; }
  template<class C> friend Tensor<C> operator*(C num, const Tensor<C>& val) { return val * num; }
  template<class C> friend Tensor<C> operator/(C num, const Tensor<C>& val) { return val / num; }

  Tensor(const T& data, const std::set<std::shared_ptr<BaseTensor<T>>>& parents) {
    ptr_ = std::make_shared<BaseTensor<T>>(data, parents);
  }
  // ptr_ contains the data. This class just manages it.
  std::shared_ptr<BaseTensor<T>> ptr_ = nullptr;

public:
  Tensor() { ptr_ = std::make_shared<BaseTensor<T>>(static_cast<T>(0)); }
  explicit Tensor(const T& data, const Activation& act) { ptr_ = std::make_shared<BaseTensor<T>>(data);}
  explicit Tensor(const T& data) { ptr_ = std::make_shared<BaseTensor<T>>(data); }
  ~Tensor() { ptr_ = nullptr; }
  Tensor(const Tensor& other) { ptr_ = other.ptr_; }
  Tensor(Tensor&& other) noexcept {
    ptr_ = other.ptr_;
    other.ptr_ = nullptr;
  }
  Tensor& operator=(const Tensor& other) {
    if (&other != this) {
      ptr_ = other.ptr_;
    }
    return *this;
  }
  Tensor& operator=(Tensor&& other) noexcept {
    ptr_ = other.ptr_;
    other.ptr_ = nullptr;
    return *this;
  }

  std::shared_ptr<BaseTensor<T>> get_ptr() const { return ptr_; }
  void set_backward(std::function<void()> func) const { ptr_->set_backward(func); }
  void backward() const { ptr_->backward(); }
  void zero_grad() { ptr_->grad_ = static_cast<T>(0); }
  void zero_grad_all() { ptr_->zero_grad_all(); }
  const T& get_data() const { return ptr_->get_data(); }
  const T& get_grad() const { return ptr_->get_grad(); }
  T& get_data() { return ptr_->get_data(); }
  T& get_grad() { return ptr_->get_grad(); }
  std::vector<BaseTensor<T>*> build_topo() const { return ptr_->build_topological_order(); }

  Tensor operator+(const Tensor& other) const {
    Tensor out(
      get_ptr()->get_data() + other.get_ptr()->get_data(),
      {get_ptr(), other.get_ptr()}
    );

    BaseTensor<T>* this_ptr = get_ptr().get();
    BaseTensor<T>* other_ptr = other.get_ptr().get();
    BaseTensor<T>* out_ptr = out.get_ptr().get();
    RegisterGradient<T>::register_backward(this_ptr, other_ptr, out_ptr, Operation::ADD);
    return out;
  }
  Tensor operator+(const T& other) const {
    auto temp = Tensor(other);
    return operator+(temp);
  }
  Tensor operator-(const Tensor& other) const {
    const auto out = Tensor(
      get_ptr()->get_data() - other.get_ptr()->get_data(),
      {get_ptr(), other.get_ptr()}
    );

    BaseTensor<T>* this_ptr = get_ptr().get();
    BaseTensor<T>* other_ptr = other.get_ptr().get();
    BaseTensor<T>* out_ptr = out.get_ptr().get();
    RegisterGradient<T>::register_backward(this_ptr, other_ptr, out_ptr, Operation::SUBTRACT);
    return out;
  }
  Tensor operator-(const T& other) const {
    const auto temp = Tensor(other);
    return operator-(temp);
  }
  Tensor operator/(const Tensor& other) const {
    auto out = Tensor(get_data() / other.get_data(),
      {get_ptr(), other.get_ptr()});
    BaseTensor<T>* this_ptr = get_ptr().get();
    BaseTensor<T>* other_ptr = other.get_ptr().get();
    BaseTensor<T>* out_ptr = out.get_ptr().get();
    RegisterGradient<T>::register_backward(this_ptr, other_ptr, out_ptr, Operation::DIVIDE);
    return out;
  }
  Tensor operator/(const T& other) const {
    const auto temp = Tensor(other);
    return operator/(temp);
  }
  Tensor operator*(const Tensor& other) const {
    auto out = Tensor(
      get_data() * other.get_data(),
      {get_ptr(), other.get_ptr()});
    BaseTensor<T>* this_ptr = get_ptr().get();
    BaseTensor<T>* other_ptr = other.get_ptr().get();
    BaseTensor<T>* out_ptr = out.get_ptr().get();
    RegisterGradient<T>::register_backward(this_ptr, other_ptr, out_ptr, Operation::MULTIPLY);
    return out;
  }
  Tensor operator*(const T& other) const {
    const auto temp = Tensor(other);
    return operator*(temp);
  }
  void operator-() {
    ptr_->get_data() = -ptr_->get_data();
  }

  Tensor activation_output(const Activation& act) const {
    auto output_data = ActivationOutput<T>::func(get_data(), act);
    auto output_tensor = Tensor(output_data, {get_ptr()});

    BaseTensor<T>* out_ptr = output_tensor.get_ptr().get();
    RegisterGradient<T>::register_backward(get_ptr().get(), out_ptr, act);

    return output_tensor;
  }
};

#endif //TENSOR_HPP

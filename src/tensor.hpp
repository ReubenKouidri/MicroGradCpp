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

  T _data { static_cast<T>(0) };
  T _grad { static_cast<T>(0) };
  std::set<std::shared_ptr<BaseTensor>> _parents;
  std::function<void()> _backward { do_nothing };

public:
  BaseTensor(const T& data, const std::set<std::shared_ptr<BaseTensor>>& parents):
    _data{data}, _parents{parents} {}
  explicit BaseTensor(const T& data): _data(data) {}
  ~BaseTensor() = default;
  BaseTensor(const BaseTensor&) = delete;
  BaseTensor(BaseTensor&&) = delete;
  BaseTensor& operator=(const BaseTensor& other) = delete;
  BaseTensor& operator=(BaseTensor&& other) = delete;

  const T& get_data() const { return _data; }
  const T& get_grad() const { return _grad; }
  T& get_data() { return _data; }
  T& get_grad() { return _grad; }
  const std::set<std::shared_ptr<BaseTensor>>& get_parent_ptrs() const { return _parents; }
  void zero_grad() { _grad = static_cast<T>(0); }
  void zero_grad_all() {
    const auto order = build_topological_order();
    for (auto n = order.rbegin(); n != order.rend(); ++n)
      (*n)->zero_grad();
  }

  void set_backward(const std::function<void()>& func) { _backward = func; }
  void step(const T& learning_rate) { _data -= learning_rate * _grad; }

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
    _grad = static_cast<T>(1); // Set dx/dx=1
    for (const auto& node: std::ranges::reverse_view(topological_order)) {
      if (node != nullptr) node->_backward();
      else { return; }
    }
  }
};


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
    _ptr = std::make_shared<BaseTensor<T>>(data, parents);
  }
  // _ptr contains the data. This class just manages it.
  std::shared_ptr<BaseTensor<T>> _ptr = nullptr;

public:
  Tensor() {_ptr = std::make_shared<BaseTensor<T>>(static_cast<T>(0)); }
  explicit Tensor(const T& data) { _ptr = std::make_shared<BaseTensor<T>>(data); }
  ~Tensor() { _ptr = nullptr; }
  Tensor(const Tensor& other) { _ptr = other._ptr; }
  Tensor(Tensor&& other) noexcept {
    _ptr = other._ptr;
    other._ptr = nullptr;
  }
  Tensor& operator=(const Tensor& other) {
    if (&other != this) _ptr = other._ptr;
    return *this;
  }
  Tensor& operator=(Tensor&& other) noexcept {
    _ptr = other._ptr;
    other._ptr = nullptr;
    return *this;
  }

  std::shared_ptr<BaseTensor<T>> get_ptr() const { return _ptr; }
  void set_backward(std::function<void()> func) const { _ptr->set_backward(func); }
  void backward() const { _ptr->backward(); }
  void zero_grad() { _ptr->_grad = static_cast<T>(0); }
  void zero_grad_all() { _ptr->zero_grad_all(); }
  const T& get_data() const { return _ptr->get_data(); }
  const T& get_grad() const { return _ptr->get_grad(); }
  T& get_data() { return _ptr->get_data(); }
  T& get_grad() { return _ptr->get_grad(); }
  std::vector<BaseTensor<T>*> build_topo() const { return _ptr->build_topological_order(); }

  Tensor operator+(const Tensor& other) const {
    Tensor out(
      get_ptr()->get_data() + other.get_ptr()->get_data(),
      {get_ptr(), other.get_ptr()}
    );

    BaseTensor<T>* this_ptr = get_ptr().get();
    BaseTensor<T>* other_ptr = other.get_ptr().get();
    BaseTensor<T>* out_ptr = out.get_ptr().get();
    RegisterGradient<T>::register_backward(this_ptr, other_ptr, out_ptr, RegisterGradient<T>::Operation::ADD);
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
    RegisterGradient<T>::register_backward(this_ptr, other_ptr, out_ptr, RegisterGradient<T>::Operation::SUBTRACT);
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
    RegisterGradient<T>::register_backward(this_ptr, other_ptr, out_ptr, RegisterGradient<T>::Operation::DIVIDE);
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
    RegisterGradient<T>::register_backward(this_ptr, other_ptr, out_ptr, RegisterGradient<T>::Operation::MULTIPLY);
    return out;
  }
  Tensor operator*(const T& other) const {
    const auto temp = Tensor(other);
    return operator*(temp);
  }
  void operator-() {
    _ptr->get_data() = -_ptr->get_data();
  }
};

#endif //TENSOR_HPP

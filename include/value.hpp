#ifndef VALUE_HPP
#define VALUE_HPP

#include "operations.hpp"
#include <cmath>
#include <functional>
#include <stack>
#include <unordered_set>
#include <vector>

using namespace ops;

const std::function do_nothing = []{ return; };

template<typename T>
class Value_ {
  template <typename C>  friend class Value;

  template<typename C>
  friend std::ostream& operator<<(std::ostream& os, const Value_<C>& val) {
    os << "Value(" << val.get_data() << ", " << val.get_grad() << ")";
    return os;
  }
  void DFS(std::vector<Value_*>& topological_order,
           std::unordered_set<Value_*>& visited_nodes) {
    std::stack<Value_*> stack;
    stack.push(this);
    while (!stack.empty()) {
      Value_* node = stack.top();
      stack.pop();

      if (!visited_nodes.insert(node).second)
        continue;

      topological_order.push_back(node);
      for (auto& parent_ptr : node->get_parent_ptrs()) {
        if (!visited_nodes.contains(parent_ptr.get())) {
          stack.push(parent_ptr.get());
        }
      }
    }
  }

  T data_ { static_cast<T>(0) };
  T grad_ { static_cast<T>(0) };
  std::vector<std::shared_ptr<Value_>> parents_;
  std::function<void()> backward_ = do_nothing;

public:
  Value_(const T& data, const std::vector<std::shared_ptr<Value_>>& parents):
    data_(data), parents_(parents) {}
  explicit Value_(const T& data): data_(data) {}
  ~Value_() = default;
  Value_(const Value_&) = delete;
  Value_(Value_&&) = delete;
  Value_& operator=(const Value_& other) = delete;
  Value_& operator=(Value_&& other) = delete;

  const T& get_data() const { return data_; }
  const T& get_grad() const { return grad_; }
  T& get_data() { return data_; }
  T& get_grad() { return grad_; }
  const std::vector<std::shared_ptr<Value_>>& get_parent_ptrs() const {
    return parents_;
  }
  void zero_grad() { grad_ = static_cast<T>(0); }
  void set_backward(const std::function<void()>& func) { backward_ = func; }
  void step(const double& learning_rate) { data_ -= learning_rate * grad_; }

  std::vector<Value_*> build_topological_order() {
    std::vector<Value_*> topological_order;
    std::unordered_set<Value_*> visited_nodes;
    DFS(topological_order, visited_nodes);
    return topological_order;
  }

  void backward() {
    auto topo = build_topological_order();
    T clip_threshold = 1; // This is a hyperparameter
    grad_ = static_cast<T>(1); // Set dx/dx=1
    for (const auto node : topo) {
      if (std::abs(node->grad_) > clip_threshold) {
        node->grad_ = std::copysign(clip_threshold, node->grad_);
      }
      node->backward_();
    }
  }
};

template<typename T>
class Value {
  template<typename C>
  friend std::ostream& operator<<(std::ostream& os, const Value<C>& val) {
    os << "Value(" << val.get_data() << ", " << val.get_grad() << ")";
    return os;
  }
  // 'right-op' overloads
  template<typename C>
  friend Value<C> operator+(C num, const Value<C>& val) { return val + num; }
  template<typename C>
  friend Value<C> operator-(C num, const Value<C>& val) { return val - num; }
  template<typename C>
  friend Value<C> operator*(C num, const Value<C>& val) { return val * num; }
  template<typename C>
  friend Value<C> operator/(C num, const Value<C>& val) { return val / num; }

  std::shared_ptr<Value_<T>> ptr_ = nullptr;

public:
  Value(const T& data, const std::vector<std::shared_ptr<Value_<T>>>& parents) {
    ptr_ = std::make_shared<Value_<T>>(data, parents);
  }
  Value() { ptr_ = std::make_shared<Value_<T>>(static_cast<T>(0)); }
  explicit Value(const T& data) { ptr_ = std::make_shared<Value_<T>>(data); }
  ~Value() { ptr_ = nullptr; }
  Value(const Value& other) { ptr_ = other.ptr_; }
  Value(Value&& other) noexcept {
    ptr_ = other.ptr_;
    other.ptr_ = nullptr;
  }

  Value& operator=(const Value& other) {
    if (&other != this)
      ptr_ = other.ptr_;
    return *this;
  }

  Value& operator=(Value&& other) noexcept {
    ptr_ = other.ptr_;
    other.ptr_ = nullptr;
    return *this;
  }

  std::shared_ptr<Value_<T>> get_ptr() const { return ptr_; }

  void set_backward(std::function<void()> func) const {
    ptr_->set_backward(func);
  }

  void backward() const { ptr_->backward(); }

  void zero_grad() { ptr_->grad_ = static_cast<T>(0); }

  T& get_data() const { return ptr_->get_data(); }

  const T& get_grad() const { return ptr_->get_grad(); }

  void set_data(const T val) { ptr_->get_data() = val; }

  T& get_grad() { return ptr_->get_grad(); }

  void step(const double& learning_rate) { ptr_->step(learning_rate); }

  std::vector<Value_<T>*> build_topo() const {
    return ptr_->build_topological_order();
  }

  Value operator+(const Value& other) const {
    auto out = Value(get_data() + other.get_data(),
                  {get_ptr(), other.get_ptr()});
    register_op<T>(this, other, out, BinaryOp::add);
    return out;
  }

  Value operator+(const T& other) const {
    auto temp = Value(other);
    return operator+(temp);
  }

  Value& operator+=(const T& other) {
    return operator+=(Value(other));
  }

  Value& operator+=(const Value& other) {
    *this = *this + other;
    return *this;
  }

  Value operator-(const Value& other) const {
    auto out = Value(get_data() - other.get_data(),
                    {get_ptr(), other.get_ptr()});
    register_op<T>(this, other, out, BinaryOp::subtract);
    return out;
  }

  Value operator-(const T& other) const {
    const auto temp = Value(other);
    return operator-(temp);
  }

  Value& operator-=(const Value& other) {
    *this = *this - other;
    return *this;
  }

  Value operator/(const Value& other) const {
    auto out = pow(other, static_cast<T>(-1));
    return operator*(out);
  }

  Value operator/(const T& other) const {
    const auto temp = Value(other);
    return operator/(temp);
  }

  Value& operator/=(const Value& other) {
    *this = *this / other;
    return *this;
  }

  Value& operator/=(const T& other) {
    auto temp = Value(other);
    return operator/=(temp);
  }

  Value operator*(const Value& other) const {
    auto result = Value(get_data() * other.get_data(),
                      {get_ptr(), other.get_ptr()});
    register_op<T>(this, other, result, BinaryOp::multiply);
    return result;
  }

  Value operator*(const T& other) const {
    const auto temp = Value(other);
    return operator*(temp);
  }

  Value& operator*=(const Value& other) {
    *this = *this * other;
    return *this;
  }

  Value& operator*=(const T& other) {
    return operator*=(Value(other));
  }

  Value operator-() {
    return operator*(static_cast<T>(-1));
  }

  bool operator>(const Value& other) const {
    return get_data() > other.get_data();
  }
  bool operator<(const Value& other) const { return !(*this > other); }
};

#endif //VALUE_HPP
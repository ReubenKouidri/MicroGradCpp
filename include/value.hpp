#ifndef VALUE_HPP
#define VALUE_HPP

#include <cmath>
#include <vector>
#include <set>
#include <functional>
#include <ranges>
#include "grad_utils.hpp"

const std::function<void()> do_nothing = [](){ return; };

template<typename T>
class Value_ {
  template <typename C> friend class Value;

  template<typename C>
  friend std::ostream& operator<<(std::ostream& os, const Value_<C>& val) {
    os << "Value(" << val.get_data() << ", " << val.get_grad() << ")";
    return os;
  }

  T data_ { static_cast<T>(0) };
  T grad_ { static_cast<T>(0) };
  std::vector<std::shared_ptr<Value_>> parents_;
  std::function<void()> backward_ = do_nothing ;

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
  const std::vector<std::shared_ptr<Value_>>& get_parent_ptrs() const { return parents_; }
  void zero_grad() { grad_ = static_cast<T>(0); }
  void set_backward(const std::function<void()>& func) { backward_ = func; }
  void step(const double& learning_rate) { data_ -= learning_rate * grad_; }

  std::vector<Value_*> build_topological_order() {
    std::vector<Value_*> topological_order;
    std::set<Value_*> visited_nodes;
    std::function<void(Value_*, std::set<Value_*>&, std::vector<Value_*>&)> build_topo_;
    build_topo_ = [&build_topo_](
      Value_* node, std::set<Value_*>& visited, std::vector<Value_*>& order) {
      if (visited.contains(node))
        return;

      visited.insert(node);
      for (auto& parent_ptr: node->get_parent_ptrs())
        build_topo_(parent_ptr.get(), visited, order);
      order.push_back(node);
    };
    build_topo_(this, visited_nodes, topological_order);
    return topological_order;
  }

  void backward() {
    auto topo = build_topological_order();
    grad_ = static_cast<T>(1); // Set dx/dx=1
    for (const auto node : std::ranges::reverse_view(topo)) {
      node->backward_();
    }
  }
};


using namespace ops;

template<typename T>
class Value {
  template<typename C> friend std::ostream& operator<<(std::ostream& os, const Value<C>& val) {
    os << "Value(" << val.get_data() << ", " << val.get_grad() << ")";
    return os;
  }
  // 'right-op' overloads
  template<typename C> friend Value<C> operator+(C num, const Value<C>& val) { return val + num; }
  template<typename C> friend Value<C> operator-(C num, const Value<C>& val) { return val - num; }
  template<typename C> friend Value<C> operator*(C num, const Value<C>& val) { return val * num; }
  template<typename C> friend Value<C> operator/(C num, const Value<C>& val) { return val / num; }

  template<typename C>
  friend Value pow(const Value& obj, const C e) {
    auto out = Value(std::pow(obj.get_data(), e), {obj.get_ptr()});
    register_op(obj, out, UnaryOp::pow, e);
    return out;
  }

  friend Value vexp(const Value& operand) {
    auto result = Value(std::exp(operand.get_data()), {operand.get_ptr()});
    register_op(operand, result, UnaryOp::exp);
    return result;
  }

  friend Value vlog(const Value& operand) {
    auto result = Value(std::log(operand.get_data()), {operand.get_ptr()});
    register_op(operand, result, UnaryOp::ln);
    return result;
  }

  friend Value relu(const Value& operand) {
    T new_data = std::max(static_cast<T>(0), operand.get_data());
    auto result = Value(new_data, {operand.get_ptr()});
    register_op(operand, result, UnaryOp::relu);
    return result;
  }

  std::shared_ptr<Value_<T>> ptr_ = nullptr;

  Value(const T& data, const std::vector<std::shared_ptr<Value_<T>>>& parents) {
    ptr_ = std::make_shared<Value_<T>>(data, parents);
  }

public:
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

  template<typename... Args>
  static void register_op(Args&&... args) {
    Register<T>::register_op(std::forward<Args>(args)...);
  }

  std::shared_ptr<Value_<T>> get_ptr() const { return ptr_; }

  void set_backward(std::function<void()> func) const { ptr_->set_backward(func); }

  void backward() const { ptr_->backward(); }

  void zero_grad() { ptr_->grad_ = static_cast<T>(0); }

  const T& get_data() const { return ptr_->get_data(); }

  const T& get_grad() const { return ptr_->get_grad(); }

  T& get_data() { return ptr_->get_data(); }

  T& get_grad() { return ptr_->get_grad(); }

  void step(const double& learning_rate) { ptr_->step(learning_rate); }

  std::vector<Value_<T>*> build_topo() const { return ptr_->build_topological_order(); }

  Value operator+(const Value& other) const {
    auto out = Value(get_data() + other.get_data(),{get_ptr(), other.get_ptr()});
    register_op(this, other, out, BinaryOp::add);
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
    auto out = Value(get_data() - other.get_data(),{get_ptr(), other.get_ptr()});
    register_op(this, other, out, BinaryOp::subtract);
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
    auto result = Value(get_data() * other.get_data(), {get_ptr(), other.get_ptr()});
    register_op(this, other, result, BinaryOp::multiply);
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

  bool operator>(const Value& other) const { return get_data() > other.get_data(); }
  bool operator<(const Value& other) const { return !(*this > other); }
};

#endif //VALUE_HPP
#ifndef VALUE_HPP
#define VALUE_HPP

#include <vector>
#include <set>
#include <functional>
#include <ranges>
#include <random>
#include "grad_utils.hpp"


inline std::function do_nothing = []{};

template<class T>
class Value_ {
  template <class C> friend class Value;

  template<typename C>
  friend std::ostream& operator<<(std::ostream& os, const Value_<C>& val) {
    os << "Value(" << val.get_data() << ", " << val.get_grad() << ")";
    return os;
  }

  T data_ { static_cast<T>(0) };
  T grad_ { static_cast<T>(0) };
  std::set<std::shared_ptr<Value_>> parents_;
  std::function<void()> backward_ { do_nothing };

public:
  Value_(const T& data, const std::set<std::shared_ptr<Value_>>& parents):
    data_{data}, parents_{parents} {}
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
  const std::set<std::shared_ptr<Value_>>& get_parent_ptrs() const { return parents_; }
  void zero_grad() { grad_ = static_cast<T>(0); }
  void zero_grad_all() {
    const auto order = build_topological_order();
    for (auto n = order.rbegin(); n != order.rend(); ++n)
      (*n)->zero_grad();
  }

  void set_backward(const std::function<void()>& func) { backward_ = func; }
  void step(const double& learning_rate) { data_ -= learning_rate * grad_; }

  std::vector<Value_*> build_topological_order() {
    std::vector<Value_*> topological_order;
    std::set<Value_*> visited_nodes;
    // declare lambda
    std::function<void(Value_*, std::set<Value_*>&, std::vector<Value_*>&)> traverse_and_build_order;
    traverse_and_build_order = [&traverse_and_build_order](
      Value_* node, std::set<Value_*>& visited, std::vector<Value_*>& order) {
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

template<class T>
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
  template<typename C> friend Value<C> tpow(const Value<C>& obj, const C e) {
    auto out = Value(std::pow(obj.get_data(), e), {obj.get_ptr()});
    Value_<C>* obj_ptr = obj.get_ptr().get();
    Value_<C>* out_ptr = out.get_ptr().get();
    RegisterGradient<C>::register_backward(obj_ptr, out_ptr, Operation::EXP, e);
    return out;
  }
  static auto rng() {
    static std::random_device rand;
    static std::mt19937 gen(rand());
    static std::uniform_real_distribution<double> dist(0, 1);
    return dist(gen);
  }
  using rand_type = decltype(rng());

  Value(const T& data, const std::set<std::shared_ptr<Value_<T>>>& parents) {
    ptr_ = std::make_shared<Value_<T>>(data, parents);
  }
  // ptr_ contains the data. This class just manages it.
  std::shared_ptr<Value_<T>> ptr_ = nullptr;

public:
  Value() { ptr_ = std::make_shared<Value_<T>>(static_cast<T>(0)); }
  explicit Value(const T& data, const Activation& act) { ptr_ = std::make_shared<Value_<T>>(data);}
  explicit Value(const T& data) { ptr_ = std::make_shared<Value_<T>>(data); }
  explicit Value(const rand_type& data) { ptr_ = std::make_shared<Value_<rand_type>>(data); }
  ~Value() { ptr_ = nullptr; }
  Value(const Value& other) { ptr_ = other.ptr_; }
  Value(Value&& other) noexcept {
    ptr_ = other.ptr_;
    other.ptr_ = nullptr;
  }
  Value& operator=(const Value& other) {
    if (&other != this) {
      ptr_ = other.ptr_;
    }
    return *this;
  }
  Value& operator=(Value&& other) noexcept {
    ptr_ = other.ptr_;
    other.ptr_ = nullptr;
    return *this;
  }

  std::shared_ptr<Value_<T>> get_ptr() const { return ptr_; }
  void set_backward(std::function<void()> func) const { ptr_->set_backward(func); }
  void backward() const { ptr_->backward(); }
  void zero_grad() { ptr_->grad_ = static_cast<T>(0); }
  void zero_grad_all() { ptr_->zero_grad_all(); }
  const T& get_data() const { return ptr_->get_data(); }
  const T& get_grad() const { return ptr_->get_grad(); }
  T& get_data() { return ptr_->get_data(); }
  T& get_grad() { return ptr_->get_grad(); }
  void step(const double& learning_rate) { ptr_->step(learning_rate); }
  std::vector<Value_<T>*> build_topo() const { return ptr_->build_topological_order(); }

  Value operator+(const Value& other) const {
    Value out(
      get_ptr()->get_data() + other.get_ptr()->get_data(),
      {get_ptr(), other.get_ptr()}
    );

    Value_<T>* this_ptr = get_ptr().get();
    Value_<T>* other_ptr = other.get_ptr().get();
    Value_<T>* out_ptr = out.get_ptr().get();
    RegisterGradient<T>::register_backward(this_ptr, other_ptr, out_ptr, Operation::ADD);
    return out;
  }
  Value operator+(const T& other) const {
    auto temp = Value(other);
    return operator+(temp);
  }
  Value operator+=(const T& other) const {
    return operator+(other);
  }
  Value operator+=(const Value& other) const {
    return operator+(other);
  }
  Value operator-(const Value& other) const {
    const auto out = Value(
      get_ptr()->get_data() - other.get_ptr()->get_data(),
      {get_ptr(), other.get_ptr()}
    );

    Value_<T>* this_ptr = get_ptr().get();
    Value_<T>* other_ptr = other.get_ptr().get();
    Value_<T>* out_ptr = out.get_ptr().get();
    RegisterGradient<T>::register_backward(this_ptr, other_ptr, out_ptr, Operation::SUBTRACT);
    return out;
  }
  Value operator-(const T& other) const {
    const auto temp = Value(other);
    return operator-(temp);
  }
  Value operator/(const Value& other) const {
    auto out = Value(get_data() / other.get_data(),
      {get_ptr(), other.get_ptr()});
    Value_<T>* this_ptr = get_ptr().get();
    Value_<T>* other_ptr = other.get_ptr().get();
    Value_<T>* out_ptr = out.get_ptr().get();
    RegisterGradient<T>::register_backward(this_ptr, other_ptr, out_ptr, Operation::DIVIDE);
    return out;
  }
  Value operator/(const T& other) const {
    const auto temp = Value(other);
    return operator/(temp);
  }
  Value operator/=(const Value& other) const {
    return operator/(other);
  }
  Value operator*(const Value& other) const {
    auto out = Value(
      get_data() * other.get_data(),
      {get_ptr(), other.get_ptr()});
    Value_<T>* this_ptr = get_ptr().get();
    Value_<T>* other_ptr = other.get_ptr().get();
    Value_<T>* out_ptr = out.get_ptr().get();
    RegisterGradient<T>::register_backward(this_ptr, other_ptr, out_ptr, Operation::MULTIPLY);
    return out;
  }
  Value operator*(const T& other) const {
    const auto temp = Value(other);
    return operator*(temp);
  }
  void operator-() {
    ptr_->get_data() = -ptr_->get_data();
  }

  Value activation_output(const Activation& act) const {
    auto output_data = ActivationOutput<T>::func(ptr_->get_data(), act);
    auto output_tensor = Value(output_data, {get_ptr()});

    Value_<T>* out_ptr = output_tensor.get_ptr().get();
    RegisterGradient<T>::register_backward(get_ptr().get(), out_ptr, act);

    return output_tensor;
  }
};

#endif // VALUE_HPP

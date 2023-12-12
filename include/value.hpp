#ifndef VALUE_HPP
#define VALUE_HPP

#include <vector>
#include <set>
#include <functional>
#include <ranges>
#include <random>
#include "grad_utils.hpp"

const std::function<void()> do_nothing = [](){ return; };

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
    const auto order = build_topological_order();
    grad_ = static_cast<T>(1); // Set dx/dx=1
    for (const auto node: std::ranges::reverse_view(order)) {
      node->backward_();
    }
  }
};


using namespace gradient_ops;

template<class T>
class Value {
  static T rng() {
    static std::random_device rand;
    static std::mt19937 gen(rand());
    static std::uniform_real_distribution<T> dist(-1, 1);
    return dist(gen);
  }

  template<typename C> friend std::ostream& operator<<(std::ostream& os, const Value<C>& val) {
    os << "Value(" << val.get_data() << ", " << val.get_grad() << ")";
    return os;
  }
  // 'right-op' overloads
  template<class C> friend Value<C> operator+(C num, const Value<C>& val) { return val + num; }
  template<class C> friend Value<C> operator-(C num, const Value<C>& val) { return val - num; }
  template<class C> friend Value<C> operator*(C num, const Value<C>& val) { return val * num; }
  template<class C> friend Value<C> operator/(C num, const Value<C>& val) { return val / num; }

  template<class C>
  friend Value<C> pow(const Value<C>& obj, const C e) {
    auto out = Value(std::pow(obj.get_data(), e), {obj.get_ptr()});
    Value_<C>* obj_ptr = obj.get_ptr().get();
    Value_<T>* out_ptr = out.get_ptr().get();

    auto back_ = [=]() {
      // std::cout << "pow\n";
      obj_ptr->get_grad() += (e * std::pow(obj_ptr->get_data(), e - static_cast<T>(1))) * out_ptr->get_grad();
    };
    out.set_backward(back_);
    // Register<C>::register_backward(obj_ptr, out, Operation::EXP, e);
    return out;
  }
  std::shared_ptr<Value_<T>> ptr_ = nullptr;

  Value(const T& data, const std::vector<std::shared_ptr<Value_<T>>>& parents) {
    ptr_ = std::make_shared<Value_<T>>(data, parents);
  }

public:
  Value() { ptr_ = std::make_shared<Value_<T>>(static_cast<T>(rng())); }
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
    auto out = Value(
      get_data() + other.get_data(),
      {get_ptr(), other.get_ptr()}
    );

    Value_<T>* this_ptr = get_ptr().get();
    Value_<T>* other_ptr = other.get_ptr().get();
    Value_<T>* out_ptr = out.get_ptr().get();

    auto backward_function = [=] {
      // std::cout << "+\n";
      this_ptr->grad_ += out_ptr->grad_;
      other_ptr->grad_ += out_ptr->grad_;
    };
    out.set_backward(backward_function);
    // Register<T>::register_backward(this_ptr, other_ptr, out, Operation::ADD);
    return out;
  }
  Value operator+(const T& other) const {
    auto temp = Value(other);
    return operator+(temp);
  }
  // Value operator+=(const T& other) const {
  //   return operator+(other);
  // }
  // Value operator+=(const Value& other) const {
  //   return operator+(other);
  // }
  Value operator-(const Value& other) const {
    auto out = Value(
      get_data() - other.get_data(),
      {get_ptr(), other.get_ptr()}
    );
    Value_<T>* this_ptr = get_ptr().get();
    Value_<T>* other_ptr = other.get_ptr().get();
    Value_<T>* out_ptr = out.get_ptr().get();

    auto backward_function = [=] {
      // TODO: CHECK FUNCS
      // std::cout << "-\n";
      this_ptr->grad_ += out_ptr->grad_;
      other_ptr->grad_ -= out_ptr->grad_;
    };
    out.set_backward(backward_function);
    // Register<T>::register_backward(this_ptr, other_ptr, out, Operation::SUBTRACT);
    return out;
  }
  Value operator-(const T& other) const {
    const auto temp = Value(other);
    return operator-(temp);
  }
  Value operator/(const Value& other) const {
    auto out = pow(other, static_cast<T>(-1));
    return operator*(out);
  }

  Value operator/(const T& other) const {
    const auto temp = Value(other);
    return operator/(temp);
  }
  // Value operator/=(const Value& other) const {
  //   return operator/(other);
  // }
  Value operator*(const Value& other) const {
    auto out = Value(
      get_data() * other.get_data(),
      {get_ptr(), other.get_ptr()});
    Value_<T>* this_ptr = get_ptr().get();
    Value_<T>* other_ptr = other.get_ptr().get();
    Value_<T>* out_ptr = out.get_ptr().get();
    auto backward_function = [=] {
      this_ptr->grad_ += other_ptr->data_ * out_ptr->grad_;
      other_ptr->grad_ += this_ptr->data_ * out_ptr->grad_;
    };
    out.set_backward(backward_function);
    // Register<T>::register_backward(this_ptr, other_ptr, out, Operation::MULTIPLY);
    return out;
  }
  // Value operator*=(const Value& other) const {
  //   return operator*(other);
  // }
  Value operator*(const T& other) const {
    const auto temp = Value(other);
    return operator*(temp);
  }
  Value operator-() {
    return operator*(static_cast<T>(-1));
  }

  Value activation_output(const Activation& act) const {
    const T output_data = ActivationOutput<T>::func(ptr_->get_data(), act);
    auto out = Value(output_data, {get_ptr()});
    Register<T>::register_backward(get_ptr().get(), out, act);
    return out;
  }
};

#endif // VALUE_HPP

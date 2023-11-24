
#ifndef MICROGRAD_VALUE_HPP
#define MICROGRAD_VALUE_HPP

#include <iostream>
#include <cmath>
#include <utility>
#include <vector>
#include <set>
#include <functional>

inline std::function<void()> do_nothing = [](){};

class Value {
  double _data {0.0};
  double _grad {0.0};
  std::set<std::shared_ptr<Value>> _parents;
  std::string op;
  std::function<void()> _backward = do_nothing;
  std::shared_ptr<Value> _ptr = std::make_shared<*this>;

public:
  explicit Value(const double data, const std::set<std::shared_ptr<Value>>& parents, std::string op = "")
      : _data(data), _parents(parents), op(std::move(op)) {}
  Value(const double data, const std::set<std::shared_ptr<Value>>& parents): _data{data}, _parents{parents} {
    op = "";
  }
  ~Value() = default;

  friend std::ostream& operator<<(std::ostream& os, const Value& value) {
    os << "Value(_data: " << value._data << ", grad: " << value.get_grad() << ")";
    return os;
  }

  friend Value pow(const Value& val, const int exp) {
    auto out = Value(std::pow(val.get_data(), exp), {val.get_ptr(), }, "**");

    const Value* val_ptr = val.get_ptr().get();
    const Value* out_ptr = out.get_ptr().get();

    auto _back = [=]() {
      val_ptr->get_grad() += (exp * std::pow(val_ptr->get_data(), exp - 1)) * out_ptr->get_grad();
    };
    out.set_backward(_back);

    return out;
  }

  [[nodiscard]] const double& get_data() const { return _data; }
  [[nodiscard]] const double& get_grad() const {return _grad; }
  [[nodiscard]] double& get_data() { return _data; }
  [[nodiscard]] double& get_grad() { return _grad; }


  [[nodiscard]] std::shared_ptr<Value> get_ptr() const { return _ptr; }
  void set_backward(const std::function<void()>& func) { _backward = func; }

  Value operator+(const Value& other) const {
    auto out = Value(
      get_data() + other.get_data(),
      {get_ptr(), other.get_ptr()},
      "+");

    Value* const other_ptr = other.get_ptr().get();
    Value* const this_ptr = get_ptr().get();
    const Value* const out_ptr = out.get_ptr().get();
    // When a new Value is created by addition, the backward function is registered capturing these three pointers
    // by value. When _backward is called, it'll operate on these.
    // Capture by ref leaves dangling references
    auto _back = [=]() {
      this_ptr->_grad += out_ptr->_grad;
      other_ptr->_grad += out_ptr->_grad;
    };
    out.set_backward(_back);

    return out;
  }

  void operator-() { this->_data = -(this->_data); }

  Value operator*(const Value& other) const {

    auto out = Value(
      get_data() * other.get_data(),
      {get_ptr(), other.get_ptr()},
      "*");

    Value* const this_ptr = get_ptr().get();
    Value* const other_ptr = other.get_ptr().get();
    const Value* const out_ptr = out.get_ptr().get();

    auto _back = [=]() {
      this_ptr->_grad += other_ptr->_data * out_ptr->_grad;
      other_ptr->_grad += this_ptr->_data * out_ptr->_grad;
    };

    out.set_backward(_back);
    return out;
  }

  void operator/(const double denom) { _data /= denom; }

  Value operator/(const Value& other) const {
    const auto temp = pow(other, -1);
    return operator*(temp);
  }
};

#endif //MICROGRAD_VALUE_HPP
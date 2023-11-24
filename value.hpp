
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
  double _data;
  double _grad {0};
  std::set<std::shared_ptr<Value>> _parents;
  std::string op;
  std::function<void()> _backward = do_nothing;
  std::shared_ptr<Value> _ptr = std::make_shared<*this>;

public:
  explicit Value(const double data, const std::set<std::shared_ptr<Value>>& parents, std::string op = "")
      : _data(data), _parents(parents), op(std::move(op)) {}
  explicit Value(const double data) : _data(data) {}
  Value(): _data(0) { _ptr = std::make_shared<Value>(_data); }
  ~Value() = default;
  Value(const double data, const std::set<std::shared_ptr<Value>>& parents): _data{data}, _parents{parents}
  {}

  [[nodiscard]] double get_data() const { return _data; }
  [[nodiscard]] double get_grad() const {return _grad; }
  [[nodiscard]] std::shared_ptr<Value> get_ptr() { return _ptr; }
  void set_backward(const std::function<void()>& func) { _backward = func; }

  Value operator+(Value& other) {
    /*
    we create a new Value object to return e.g. for v = v1 + v2
    const_cast casts 'this' to a pointer to Value, so it can be added to the set _parents values
    we then set _backward to a lambda function that captures 'other', 'this' and 'out'
    and sets the grads of 'this' and 'other'

      z = x + y
      dz/dx = 1, dz/dy = 1
      df(z)/dx = (df/dz)(dz/dx) = (df/dz) * 1 = df/dz

      hence, 'this'-> _grad = out->_grad
      same applied for 'other'
    */

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

  Value operator*(Value other) {
    /*
      z = x * y
      dz/dx = y, dz/dy = x
      df(z)/dx = (df/dz)(dz/dx) = (df/dz) * y = out._grad * other._data
      df(z)/dy = (df/dz)(dz/dy) = (df/dz) * x = out._grad * this._data
    */

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

  friend std::ostream& operator<<(std::ostream& os, const Value& value) {
    os << "Value(_data: " << value._data << ", grad: " << value.get_grad() << ")";
    return os;
  }
};

#endif //MICROGRAD_VALUE_HPP
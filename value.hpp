
#ifndef MICROGRAD_VALUE_HPP
#define MICROGRAD_VALUE_HPP

#include <iostream>
#include <cmath>
#include <vector>
#include <set>
#include <functional>


class Value {
public:
  double data;
  double grad;
  std::set<Value*> prev;
  std::string op;
  std::function<void()> _backward;

  explicit Value(double data, const std::set<Value*>& children = {}, std::string op = "")
      : data(data), grad(0), prev(children), op(std::move(op)) {}

  Value operator+(Value other) {
    /*
    we create a new Value object to return e.g. for v = v1 + v2
    const_cast casts 'this' to a pointer to Value, so it can be added to the set previous values
    we then set _backward to a lambda function that captures 'other', 'this' and 'out'
    and sets the grads of 'this' and 'other'

      z = x + y
      dz/dx = 1, dz/dy = 1
      df(z)/dx = (df/dz)(dz/dx) = (df/dz) * 1 = df/dz

      hence, 'this'-> grad = out->grad
      same applied for 'other'
    */

    auto out = Value(data + other.data, {const_cast<Value*>(this), &other}, "+");
    out._backward = [&other, &out, this]() {
      this->grad += out.grad;
      other.grad += out.grad;
    };

    return out;
  }

  void operator-() { this->data = -(this->data); }

  Value operator*(Value other) {
    /*
      z = x * y
      dz/dx = y, dz/dy = x
      df(z)/dx = (df/dz)(dz/dx) = (df/dz) * y = out.grad * other.data
      df(z)/dy = (df/dz)(dz/dy) = (df/dz) * x = out.grad * this.data
    */
    auto out = Value(data * other.data, {const_cast<Value*>(this), &other}, "*");
    out._backward = [&other, &out, this]() {
      this->grad += other.data * out.grad;
      other.grad += this->data * out.grad;
    };

    return out;
  }

  friend std::ostream& operator<<(std::ostream& os, const Value& value) {
    os << "Value(data: " << value.data << ", grad: " << value.grad << ")";
    return os;
  }
};

#endif //MICROGRAD_VALUE_HPP

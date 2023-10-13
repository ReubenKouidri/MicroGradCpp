#include <iostream>
#include <cmath>
#include <utility>
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
      grad += out.grad;
      other.grad += out.grad;
    };

    return out;
  }

  Value operator*(Value other) {
    /*
      z = x * y
      dz/dx = y, dz/dy = x
      df(z)/dx = (df/dz)(dz/dx) = (df/dz) * y = out.grad * other.data
      df(z)/dy = (df/dz)(dz/dy) = (df/dz) * x = out.grad * this.data
     */
    auto out = Value(data * other.data, {const_cast<Value*>(this), &other}, "*");
    out._backward = [&other, &out, this]() {
      grad += other.data * out.grad;
      other.grad += data * out.grad;
    };

    return out;
  }

  friend std::ostream& operator<<(std::ostream& os, const Value& value) {
    os << "Value(data: " << value.data << ", grad: " << value.grad << ")";
    return os;
  }
};

//
//
//  Value operator-(const Value& other) const {
//    return Value(data - other.data, {const_cast<Value*>(this), const_cast<Value*>(&other)}, "-");
//  }
//
//  Value pow(double power) const {
//    double result = std::pow(data, power);
//    return Value(result, {const_cast<Value*>(this)}, "**" + std::to_string(power));
//  }
//
//  Value tanh() const {
//    double t = std::tanh(data);
//    return Value(t, {const_cast<Value*>(this)}, "tanh");
//  }
//
//  void backward() {
//    std::vector<Value*> topo;
//    std::set<Value*> visited;
//
//    std::function<void(Value*)> buildTopo = [&](Value* root) {
//      if (visited.find(root) == visited.end()) {
//        visited.insert(root);
//        for (Value* child : root->prev) {
//          buildTopo(child);
//        }
//        topo.push_back(root);
//      }
//    };
//
//    buildTopo(this);
//    grad = 1.0;
//    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
//      Value* node = *it;
//
//      if (node->op == "+") {
//        node->prev[0]->grad += grad;
//        node->prev[1]->grad += grad;
//      } else if (node->op == "*") {
//        node->prev[0]->grad += node->prev[1]->data * grad;
//        node->prev[1]->grad += node->prev[0]->data * grad;
//      } else if (node->op == "-") {
//        node->prev[0]->grad -= grad;
//      } else if (node->op.substr(0, 2) == "**") {
//        double power = std::stod(node->op.substr(2));
//        node->prev[0]->grad += power * std::pow(node->prev[0]->data, power - 1) * grad;
//      } else if (node->op == "tanh") {
//        double t = std::tanh(node->data);
//        node->prev[0]->grad += (1 - t * t) * grad;
//      }
//    }
//  }
//};

int main() {
  Value a(1.0);
  Value b(2.0);
  Value c(3.0);
  Value d(4.0);

  // set grads
  a.grad = 1.0;
  b.grad = 1.0;

  auto loss_func = [&]() -> double {
    return std::pow((a.data - b.data), 2);
  };

  auto loss = Value(loss_func(), {&a, &b});
  loss.grad = 1.0;
  std::cout << "loss: " << loss << std::endl;
  loss._backward;
  // Perform backward propagation
  auto z = a + b;
  z = z * c;
  z._backward();

  std::cout << a  << std::endl;
  std::cout << b << std::endl;
  std::cout << z << std::endl;

  return 0;
}
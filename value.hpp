
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
  std::vector<Value*> prev;
  std::string op;

  Value(double data, const std::vector<Value*>& children = {}, const std::string& op = "")
      : data(data), grad(0), prev(children), op(op) {}

  Value operator+(const Value& other) const {
    return Value(data + other.data, {const_cast<Value*>(this), const_cast<Value*>(&other)}, "+");
  }

  Value operator*(const Value& other) const {
    return Value(data * other.data, {const_cast<Value*>(this), const_cast<Value*>(&other)}, "*");
  }

  Value operator-(const Value& other) const {
    return Value(data - other.data, {const_cast<Value*>(this), const_cast<Value*>(&other)}, "-");
  }

  Value pow(double power) const {
    double result = std::pow(data, power);
    return Value(result, {const_cast<Value*>(this)}, "**" + std::to_string(power));
  }

  Value tanh() const {
    double t = std::tanh(data);
    return Value(t, {const_cast<Value*>(this)}, "tanh");
  }

  void backward() {
    std::vector<Value*> topo;
    std::set<Value*> visited;

    std::function<void(Value*)> buildTopo = [&](Value* root) {
      if (visited.find(root) == visited.end()) {
        visited.insert(root);
        for (Value* child : root->prev) {
          buildTopo(child);
        }
        topo.push_back(root);
      }
    };

    buildTopo(this);
    grad = 1.0;
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
      Value* node = *it;
      if (node->op == "+") {
        node->prev[0]->grad += grad;
        node->prev[1]->grad += grad;
      } else if (node->op == "*") {
        node->prev[0]->grad += node->prev[1]->data * grad;
        node->prev[1]->grad += node->prev[0]->data * grad;
      } else if (node->op == "-") {
        node->prev[0]->grad -= grad;
      } else if (node->op.substr(0, 2) == "**") {
        double power = std::stod(node->op.substr(2));
        node->prev[0]->grad += power * std::pow(node->prev[0]->data, power - 1) * grad;
      } else if (node->op == "tanh") {
        double t = std::tanh(node->data);
        node->prev[0]->grad += (1 - t * t) * grad;
      }
    }
  }
};

int main() {
  Value x(2.0);
  Value y(3.0);
  Value z = x * y + x.pow(2) - y;

  // Perform backward propagation
  z.backward();

  std::cout << "x.grad: " << x.grad << std::endl; // Output: 7.62149
  std::cout << "y.grad: " << y.grad << std::endl; // Output: -1.07077

  return 0;
}


#endif //MICROGRAD_VALUE_HPP

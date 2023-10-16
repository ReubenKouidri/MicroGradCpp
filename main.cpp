#include <iostream>
#include <cmath>
#include <ranges>
#include <utility>
#include <vector>
#include <set>
#include <unordered_set>
#include <functional>
#include <algorithm>


class Value {
public:
  double data;
  double grad;
  std::set<Value*> prev;
  std::string op;
  std::function<void()> _backward;

  explicit Value(double data, const std::set<Value*>& children = {}, std::string op = "")
      : data(data), grad(0), prev(children), op(std::move(op)) {
    _backward = []() {};
  }

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

  void operator-() { this->data = -(this->data); }

  Value operator-(Value& other) {
    /*
     * z = x - y
     * dz/dx = 1, dz/dy = -1
     * df(z)/dx = (df/dz)(dz/dx) = (df/dz) * 1 = df/dz
     * df(z)/dy = (df/dz)(dz/dy) = (df/dz) * -1 = -df/dz
     */
    auto out = Value(data - other.data, {const_cast<Value*>(this), const_cast<Value*>(&other)}, "-");
    out._backward = [&other, &out, this]() {
      this->grad += out.grad;
      other.grad -= out.grad;
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

    auto out = Value(data * other.data, {const_cast<Value*>(this), &other} , "*");
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

  void backward() {
    auto topo = std::vector<Value*>();  // topological order
    auto visited = std::unordered_set<Value*>();  // keep track of unique_visited nodes; no duplicates

    std::function<void(Value*)> build_topo = [&](Value* root) {
      if (!visited.count(root)) {
        visited.insert(root);
        for (auto child: root->prev)  // for each child of root
          build_topo(child);  // recurse
        topo.emplace_back(root);  // this might be needed instead of the above
      }
    };

    build_topo(this);
    this->grad = 1.0;  // set grad of loss to zero

    std::reverse(topo.begin(), topo.end());
    for (auto v: topo) {
      std::cout << *v << '\n';
      v->_backward();
      std::cout << *v << '\n';
    }
  }

};

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
  Value a(0.5);
  Value b(0.6);
  auto z = a * b;
  z.backward();

//  std::cout << a << std::endl;
//  std::cout << b << std::endl;
//  std::cout << z << std::endl;
//
//  Value c(3.0);
//  Value d(4.0);
//  std::cout << c << std::endl;
//  -c;
//  std::cout << c << std::endl;
  // set grads
  std::cout << "z: " << z << '\n';
  std::cout << "a: " << a << '\n';
  std::cout << "b: " << b << '\n';
  // Perform backward propagation


  return 0;
}
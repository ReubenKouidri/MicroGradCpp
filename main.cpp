#include <iostream>
#include <cmath>
#include <ranges>
#include <utility>
#include <vector>
#include <set>
#include <unordered_set>
#include <functional>
#include <algorithm>
#include <random>


class Value {
public:
  double data;
  double grad;
  std::set<Value*> prev;
  std::string op;
  std::function<void()> _backward;

  explicit Value(double data = 0.0,
                 const std::set<Value*>& children = {},
                 std::string op = "")
      : data(data), grad(0), prev(children), op(std::move(op)) {
    _backward = []() {};
  }

  Value operator+(Value& other) {
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

    auto out = Value(data + other.data,
                     {const_cast<Value*>(this), const_cast<Value*>(&other)},
                     "+");

    out._backward = [&other, &out, this]() {
      grad += out.grad;
      other.grad += out.grad;
    };

    return out;
  }

  Value operator+=(Value&& other) {
    auto new_data = data + other.data;
    auto out = Value(new_data,
                     {const_cast<Value*>(this), const_cast<Value*>(&other)},
                     "+");
    out._backward = [&other, &out, this]() mutable {
      this->grad += out.grad;
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

  Value operator*(Value& other) {
    /*
      z = x * y
      dz/dx = y, dz/dy = x
      df(z)/dx = (df/dz)(dz/dx) = (df/dz) * y = out.grad * other.data
      df(z)/dy = (df/dz)(dz/dy) = (df/dz) * x = out.grad * this.data
    */

    auto out = Value(data * other.data, {const_cast<Value*>(this), const_cast<Value*>(&other)}, "*");
    out._backward = [&other, &out, this]() {
      this->grad += other.data * out.grad;
      other.grad += this->data * out.grad;
    };

    return out;
  }

  Value pow(const int n) {
    auto out = Value(std::pow(data, n),
                     {const_cast<Value*>(this)},
                     "**" + std::to_string(n)
    );

    out._backward = [&out, this, n]() {
      this->grad += (n * std::pow(data, n - 1)) * out.grad;
    };

    return out;
  }

  auto register_tanh_grad(double t, Value& out) {
    grad += (1 - std::pow(t, 2)) * out.grad;
  }

  Value tanh() {
    auto t = std::tanh(this->data);
    auto out = Value(t, {const_cast<Value*>(this)}, "tanh");
    register_tanh_grad(t, out);
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
        for (const auto child: root->prev)  // for each child of root
          build_topo(child);  // recurse
        topo.emplace_back(root);  // this might be needed instead of the above
      }
    };

    build_topo(this);
    this->grad = 1.0;  // set grad of loss to zero

    std::reverse(topo.begin(), topo.end());
    for (auto v: topo) {
      v->_backward();
    }
  }

};


double frand() {
  // static so that only initialised once and reused for all function calls
  static std::mt19937_64 rng(std::random_device{}());
  static std::uniform_real_distribution<double> dist(0.0, 1.0);
  return dist(rng);
}


class Neuron {
public:
  std::vector<Value> weights;
  Value bias;

  explicit Neuron(size_t nin) : bias(0.0) {
    for (size_t i = 0; i < nin; i++) {
      weights.emplace_back(frand());
    }
  }

  Value operator()(std::vector<Value>& inputs) {
    auto out = Value();
    for (int i = 0; i < inputs.size(); i++) {
      out += inputs[i] * weights[i];
    }
    out = out.tanh();
    return out;
  }

  [[nodiscard]]
  const std::vector<Value>& parameters() const {
    return weights;
  }
};


class Layer {
public:
  std::vector<Neuron> neurons;
  explicit Layer(int nin, int nout) {
    for (int i = 0; i < nout; i++) {
      neurons.emplace_back(nin);
    }
  }

  auto parameters() const {
    std::vector<Value> p;
    for (const auto& n: neurons) {
      for (const auto& w: n.parameters()) {
        p.emplace_back(w);
      }
    }
    return p;
  }


  std::vector<Value> operator()(std::vector<Value>& inputs) {
    std::vector<Value> out;
    for (auto& n: neurons) {
      out.emplace_back() = n(inputs);
    }
    return out;
  }
};


class MLP {
public:
  std::vector<Layer> layers;
  MLP(int nin, std::vector<int> nouts) {
    auto sz = nouts.insert(nouts.begin(), nin);
    for (int i = 0; i < nouts.size(); i++) {
      layers.emplace_back(sz[i], sz[i + 1]);
    }
  }
  std::vector<Value> operator()(std::vector<Value>& inputs) {
    auto out = inputs;
    for (auto& l: layers) {
      out = l(out);
    }
    return out;
  }

  std::vector<Value> parameters() {
    std::vector<Value> p;
    for (const auto& l: layers) {
      for (const auto& n: l.neurons) {
        for (const auto& param: n.parameters()) {
          p.emplace_back(param);
        }
      }
    }
    return p;
  }
};

int main() {
  std::vector<int> nouts = {4, 4, 1};
  int nin = 3;
  std::vector<Value> input = {Value(1.0), Value(2.0), Value(3.0)};
  std::vector<double> targets = {1.0, -1.0, 1.0};

  auto mlp = MLP(nin, nouts);
  auto preds = mlp(input);

//  Value a(0.5);
//  Value b(0.6);
//  Value c(2.0);
//  auto z = a * b;
//  auto x = z - c;
//  x.backward();
//
//
//  std::cout << "z: " << z << '\n';
//  std::cout << "a: " << a << '\n';
//  std::cout << "b: " << b << '\n';
//  std::cout << "c: " << c << '\n';
//  std::cout << "x: " << x << '\n';

  return 0;
}
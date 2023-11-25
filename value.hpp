
#ifndef MICROGRAD_VALUE_HPP
#define MICROGRAD_VALUE_HPP

#include <iostream>
#include <cmath>
#include <ranges>
#include <utility>
#include <vector>
#include <set>
#include <functional>
// ts: 18:14 to return to

inline std::function<void()> do_nothing = [](){};

class Value {
  double _data {0.0};
  double _grad {0.0};
  std::set<std::shared_ptr<Value>> _parents;
  std::string op;
  std::function<void()> _backward = do_nothing;

public:
  explicit Value(const double data, const std::set<std::shared_ptr<Value>>& parents, std::string op = "")
      : _data(data), _parents(parents), op(std::move(op)) {}
  explicit Value(const double data): _data(data) {}
  Value(const double data, const std::set<std::shared_ptr<Value>>& parents): _data{data}, _parents{parents} { op = ""; }
  ~Value() = default;

  friend std::ostream& operator<<(std::ostream& os, const Value& value) {
    os << "Value(_data: " << value._data << ", grad: " << value.get_grad() << ")";
    return os;
  }

  [[nodiscard]] const double& get_data() const { return _data; }
  [[nodiscard]] double& get_data() {return _data; }
  void set_data(const double new_val) { _data = new_val; }
  void set_grad(const double new_val) { _grad = new_val; }
  [[nodiscard]] const double& get_grad() const {return _grad; }
  [[nodiscard]] double& get_grad() { return _grad; }
  void zero_grad() { _grad = 0.0; }
  void zero_grad_all() {
    const auto order = build_topological_order();
    for (const auto& n : std::ranges::reverse_view(order)) { n->zero_grad(); }
  }

  [[nodiscard]] const std::set<std::shared_ptr<Value>>& get_parents() const { return _parents; }
  void set_backward(const std::function<void()>& func) { _backward = func; }

  Value operator+(Value& other) {
    auto out = Value(get_data() + other.get_data(),
    {std::make_shared<Value>(*this), std::make_shared<Value>(other)}, "+");

    auto out_ptr = &out;
    auto other_ptr = &other;

    auto _back = [this, out_ptr, other_ptr]() mutable {
      this->_grad += out_ptr->_grad;
      other_ptr->_grad += out_ptr->_grad;
    };
    out.set_backward(_back);
    return out;
  }

  Value operator+(const double other) {
    auto temp = Value(other);
    return operator+(temp);
  }

  void operator-() { _data = -_data; }

  Value operator*(Value& other) {

    auto out = Value(
      get_data() * other.get_data(),
      {std::make_shared<Value>(*this), std::make_shared<Value>(other)},
      "*");

    // Value* const this_ptr = get_ptr().get();
    // Value* const other_ptr = other.get_ptr().get();
    // const Value* const out_ptr = out.get_ptr().get();
    auto out_ptr = &out;
    auto other_ptr = &other;

    auto _back = [this, out_ptr, other_ptr]() mutable {
      this->_grad += other_ptr->_data * out_ptr->_grad;
      other_ptr->_grad += this->_data * out_ptr->_grad;
    };

    out.set_backward(_back);
    return out;
  }

  Value operator*(const double other) {
    auto temp = Value(other);
    return operator*(temp);
  }



  /*
  Value operator/(const Value& other) const {
    const auto temp = pow(other, -1);
    return operator*(temp);
  }

  Value operator/(const double other) const {
    const auto temp = Value(other);
    return pow(temp, -1);
  }
  */

  std::vector<Value*> build_topological_order() {

    std::vector<Value*> topological_order;
    std::set<Value*> visited_nodes;
    // declare lambda
    std::function<void(Value*, std::set<Value*>&, std::vector<Value*>&)> traverse_and_build_order;
    traverse_and_build_order = [&traverse_and_build_order](
      Value* node, std::set<Value*>& visited, std::vector<Value*>& order)
    {
      if (visited.contains(node))
        return;

      visited.insert(node);
      for (const auto& parent_ptr : node->get_parents())
        traverse_and_build_order(parent_ptr.get(), visited, order);
      order.push_back(node);
    };

    traverse_and_build_order(this, visited_nodes, topological_order);
    return topological_order;
  }

  void backward() {
    auto topological_order = build_topological_order();
    _grad = 1.0; // Set dx/dx=1
    for (const auto& node : std::ranges::reverse_view(topological_order)) {
      if (node != nullptr) node->_backward();
      else { return; }
    }
  }
};

#endif //MICROGRAD_VALUE_HPP
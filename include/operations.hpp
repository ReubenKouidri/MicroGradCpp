#ifndef OPERATIONS_HPP
#define OPERATIONS_HPP

#include <iostream>

template <typename T>
class Value_;

template <typename T>
class Value;

namespace ops {

enum class BinaryOp {
  add,
  subtract,
  multiply,
  divide,
};

enum class UnaryOp {
  tanh,
  relu,
  softmax,
  exp,
  ln,
  pow,
};

template <typename T>
inline void register_op(const Value<T> &operand,
                        Value<T> &result,
                        const UnaryOp &op) {
  Value_<T> *operand_ptr = operand.get_ptr().get();
  Value_<T> *result_ptr = result.get_ptr().get();

  std::function<void()> backward_function;
  switch (op) {
    case UnaryOp::exp:
      backward_function = [operand_ptr, result_ptr] {
        operand_ptr->get_grad() +=
            result_ptr->get_grad()*result_ptr->get_data();
      };
      break;
    case UnaryOp::ln:
      backward_function = [operand_ptr, result_ptr] {
        operand_ptr->get_grad() +=
            result_ptr->get_grad()/operand_ptr->get_data();
      };
      break;
    case UnaryOp::relu:
      backward_function = [operand_ptr, result_ptr] {
        if (operand_ptr->get_data() > static_cast<T>(0))
          operand_ptr->get_grad() += result_ptr->get_grad();
      };
      break;
    case UnaryOp::tanh:
      backward_function = [operand_ptr, result_ptr] {
        operand_ptr->get_grad() +=
            (1 - std::pow(result_ptr->get_data(), 2))*
                result_ptr->get_grad();
      };
      break;
    default:std::cout << "Error registering exp backward func\n";
      break;
  }
  result.set_backward(backward_function);
}

template <typename T>
inline void register_op(const Value<T> &operand,
                        Value<T> &result,
                        const UnaryOp &op,
                        const int e) {
  Value_<T> *result_ptr = result.get_ptr().get();
  Value_<T> *operand_ptr = operand.get_ptr().get();

  std::function<void()> backward_function;
  switch (op) {
    case UnaryOp::pow:
      backward_function = [operand_ptr, result_ptr, e] {
        operand_ptr->get_grad() +=
            e*std::pow(operand_ptr->get_data(), e - static_cast<T>(1))*
                result_ptr->get_grad();
      };
      break;
    default:std::cout << "Error registering exp backward func\n";
  }
  result.set_backward(backward_function);
}

template <typename T>
inline void register_op(const Value<T> *left_operand,
                        const Value<T> &right_operand,
                        Value<T> &result,
                        const BinaryOp &op) {
  Value_<T> *result_ptr = result.get_ptr().get();
  Value_<T> *left_ptr = left_operand->get_ptr().get();
  Value_<T> *right_ptr = right_operand.get_ptr().get();

  std::function<void()> backward_function;
  switch (op) {
    case BinaryOp::add:
      backward_function = [left_ptr, right_ptr, result_ptr] {
        left_ptr->get_grad() += result_ptr->get_grad();
        right_ptr->get_grad() += result_ptr->get_grad();
      };
      break;
    case BinaryOp::subtract:
      backward_function = [left_ptr, right_ptr, result_ptr] {
        left_ptr->get_grad() += result_ptr->get_grad();
        right_ptr->get_grad() -= result_ptr->get_grad();
      };
      break;
    case BinaryOp::multiply:
      backward_function = [left_ptr, right_ptr, result_ptr] {
        left_ptr->get_grad() +=
            right_ptr->get_data()*result_ptr->get_grad();
        right_ptr->get_grad() +=
            left_ptr->get_data()*result_ptr->get_grad();
      };
      break;
    case BinaryOp::divide:
      backward_function = [left_ptr, right_ptr, result_ptr] {
        left_ptr->get_grad() +=
            result_ptr->get_grad()/right_ptr->get_data();
        right_ptr->get_grad() +=
            -left_ptr->get_data()*
                result_ptr->get_grad()/std::pow(right_ptr->get_data(), 2);
      };
      break;
    default:
      std::cout << "WARNING! Setting default backward_function - [](){}\n";
      break;
  }
  result.set_backward(backward_function);
}

template <typename T>
inline Value<T> exp(const Value<T> &operand) {
  auto result = Value(std::exp(operand.get_data()), {operand.get_ptr()});
  register_op<T>(operand, result, UnaryOp::exp);
  return result;
}

template <typename T, typename C>
inline Value<T> pow(const Value<T> &obj, const C e) {
  auto out = Value(std::pow(obj.get_data(), e), {obj.get_ptr()});
  register_op<T>(obj, out, UnaryOp::pow, e);
  return out;
}

template <typename T>
inline Value<T> log(const Value<T> &operand) {
  auto result = Value(std::log(operand.get_data()), {operand.get_ptr()});
  register_op<T>(operand, result, UnaryOp::ln);
  return result;
}

template <typename T>
inline Value<T> relu(const Value<T> &operand) {
  T new_data = std::max(static_cast<T>(0), operand.get_data());
  auto result = Value(new_data, {operand.get_ptr()});
  register_op<T>(operand, result, UnaryOp::relu);
  return result;
}
}; // namespace ops

#endif //OPERATIONS_HPP

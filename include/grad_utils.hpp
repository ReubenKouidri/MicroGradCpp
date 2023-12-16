#ifndef GRAD_UTILS_HPP
#define GRAD_UTILS_HPP

#include <iostream>

template<typename T> class Value_;
template<typename T> class Value;

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
    sigmoid,
    exp,
    ln,
    pow,
  };

  template<typename T>
  class Register {
  public:
    static void register_op(const Value<T>& operand, Value<T>& result, const UnaryOp& op) {
      Value_<T>* operand_ptr = operand.get_ptr().get();
      Value_<T>* result_ptr = result.get_ptr().get();

      std::function<void()> backward_function;
      switch (op) {
        case UnaryOp::exp:
          backward_function = [operand_ptr, result_ptr] {
            operand_ptr->get_grad() += result_ptr->get_grad() * result_ptr->get_data();
          };
          break;
        case UnaryOp::ln:
          backward_function = [operand_ptr, result_ptr] {
            operand_ptr->get_grad() += result_ptr->get_grad() / operand_ptr->get_data();
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
            operand_ptr->get_grad() += (1 - std::pow(result_ptr->get_data(), 2)) * result_ptr->get_grad();
          };
          break;
        default:
          std::cout << "Error registering exp backward func\n";
          break;
      }
      result.set_backward(backward_function);
    }

    static void register_op(const Value<T>& operand, Value<T>& result, const UnaryOp& op, const int e) {
      Value_<T>* result_ptr = result.get_ptr().get();
      Value_<T>* operand_ptr = operand.get_ptr().get();

      std::function<void()> backward_function;
      switch (op) {
        case UnaryOp::pow:
          backward_function = [operand_ptr, result_ptr, e] {
            operand_ptr->get_grad() += (e * std::pow(operand_ptr->get_data(), e - static_cast<T>(1))) * result_ptr->get_grad();
          };
          break;
        default:
          std::cout << "Error registering exp backward func\n";
      }
      result.set_backward(backward_function);
    }

    static void register_op(const Value<T>* left_operand, const Value<T>& other, Value<T>& out, const BinaryOp& op) {
      Value_<T>* out_ptr = out.get_ptr().get();
      Value_<T>* this_ptr = left_operand->get_ptr().get();
      Value_<T>* other_ptr = other.get_ptr().get();

      std::function<void()> backward_function;
      switch (op) {
        case BinaryOp::add:
          backward_function = [this_ptr, other_ptr, out_ptr] {
            this_ptr->get_grad() += out_ptr->get_grad();
            other_ptr->get_grad() += out_ptr->get_grad();
          };
          break;
        case BinaryOp::subtract:
          backward_function = [this_ptr, other_ptr, out_ptr] {
            this_ptr->get_grad() += out_ptr->get_grad();
            other_ptr->get_grad() -= out_ptr->get_grad();
          };
          break;
        case BinaryOp::multiply:
          backward_function = [this_ptr, other_ptr, out_ptr] {
            this_ptr->get_grad() += other_ptr->get_data() * out_ptr->get_grad();
            other_ptr->get_grad() += this_ptr->get_data() * out_ptr->get_grad();
          };
          break;
        case BinaryOp::divide:
          backward_function = [this_ptr, other_ptr, out_ptr] {
            this_ptr->get_grad() += out_ptr->get_grad() / other_ptr->get_data();
            other_ptr->get_grad() += -1 * this_ptr->get_data() * out_ptr->get_grad() / std::pow(other_ptr->get_data(), 2);
          };
          break;
        default:
          std::cout << "WARNING! Setting default backward_function - [](){}\n";
          break;
      }
      out.set_backward(backward_function);
    }
  };
} // namespace ops

#endif //GRAD_UTILS_HPP

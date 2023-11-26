#ifndef GRAD_UTILS_HPP
#define GRAD_UTILS_HPP

#include "tensor.hpp"

template<typename T>
class BaseTensor;


template<typename T>
class RegisterGradient {
public:
  enum class Operation {
    ADD,
    SUBTRACT,
    MULTIPLY,
    DIVIDE,
  };

  static void register_backward(BaseTensor<T>* this_ptr, BaseTensor<T>* other_ptr, BaseTensor<T>* out_ptr,
                                const Operation& op) {
    std::function<void()> backward_function;
    switch (op) {
      case Operation::ADD:
        backward_function = [this_ptr, other_ptr, out_ptr]() {
          this_ptr->get_grad() += out_ptr->get_grad();
          other_ptr->get_grad() += out_ptr->get_grad();
        };
        break;
      case Operation::SUBTRACT:
        backward_function = [this_ptr, other_ptr, out_ptr]() {
          this_ptr->get_grad() += out_ptr->get_grad();
          other_ptr->get_grad() -= out_ptr->get_grad();
        };
        break;
      case Operation::MULTIPLY:
        backward_function = [this_ptr, other_ptr, out_ptr] {
          this_ptr->get_grad() += other_ptr->get_data() * out_ptr->get_grad();
          other_ptr->get_grad() += this_ptr->get_data() * out_ptr->get_grad();
        };
        break;
      case Operation::DIVIDE:
        backward_function = [this_ptr, other_ptr, out_ptr] {
          this_ptr->get_grad() += out_ptr->get_grad() / other_ptr->get_data();
          other_ptr->get_grad() += -1 * this_ptr->get_data() * out_ptr->get_grad() / std::pow(other_ptr->get_data(), 2);
        };
        break;
      default:
        // Default to do nothing if operation is not recognized
        std::cout << "WARNING! Setting default backward_function - [](){}\n";
        backward_function = []() {
        };
        break;
    }
    out_ptr->set_backward(backward_function);
  }
};

#endif //GRAD_UTILS_HPP

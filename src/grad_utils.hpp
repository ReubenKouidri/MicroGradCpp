#ifndef GRAD_UTILS_HPP
#define GRAD_UTILS_HPP

template<typename T> class Value_;
// template<typename T> class Value;

// An activation function acts on this_ptr's data and returns an output tensor
// It registers the gradient function of the output

namespace gradient_ops {
  enum class Operation {
    ADD,
    SUBTRACT,
    MULTIPLY,
    DIVIDE,
    EXP,
  };

  enum class Activation {
    TANH,
    RELU,
    SIGMOID,
  };

  template<typename T>
  class ActivationOutput {
  public:
    static T func(const T data, const Activation& act) {
      switch (act) {
        case Activation::RELU:
          return std::max(data, static_cast<T>(0));
        case Activation::TANH:
          return std::tanh(data);
        default:
          std::cout << "No activaiton\n";
          return static_cast<T>(0);
      }
    }
  };

  template<typename T>
  class RegisterGradient {
  public:
    static void register_backward(Value_<T>* this_ptr, Value_<T>* out_ptr, const Activation& act) {
      std::function<void()> backward_function = []{};
      switch (act) {
        case Activation::RELU:
          backward_function = [this_ptr, out_ptr] {
            this_ptr->get_grad() += out_ptr->get_data();
          };
          break;
        case Activation::TANH:
          backward_function = [this_ptr, out_ptr] {
            this_ptr->get_grad() += (1 - std::pow(out_ptr->get_data(), 2)) * out_ptr->get_grad();
          };
          break;
        default:
          std::cout <<  "WARNING! No activation set\n";
          break;
      }
      out_ptr->set_backward(backward_function);
    }
    static void register_backward(Value_<T>* obj_ptr, Value_<T>* out_ptr, const Operation& op, const int e) {
      std::function<void()> backward_function = []{};
      switch (op) {
        case Operation::EXP:
          backward_function = [obj_ptr, out_ptr, e] {
            obj_ptr->get_grad() += out_ptr->get_grad() * e * std::pow(obj_ptr->get_data(), e - 1);
          };
        break;
        default:
          std::cout << "Error registering exp backward func\n";
        out_ptr->set_backward(backward_function);
      }
    }

    static void register_backward(Value_<T>* this_ptr, Value_<T>* other_ptr, Value_<T>* out_ptr,
                                  const Operation& op) {
      std::function<void()> backward_function = []{};
      switch (op) {
        case Operation::ADD:
          backward_function = [this_ptr, other_ptr, out_ptr] {
            this_ptr->get_grad() += out_ptr->get_grad();
            other_ptr->get_grad() += out_ptr->get_grad();
          };
          break;
        case Operation::SUBTRACT:
          backward_function = [this_ptr, other_ptr, out_ptr] {
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
          std::cout << "WARNING! Setting default backward_function - [](){}\n";
          break;
      }
      out_ptr->set_backward(backward_function);
    }
  };
} // namespace gradient_ops

#endif //GRAD_UTILS_HPP

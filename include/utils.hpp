#ifndef UTILS_HPP
#define UTILS_HPP

#include <iostream>
#include <vector>
#include "data.hpp"

template <typename T>
class MLP;

inline void visualise_input(const std::vector<double>& input) {
  for (size_t i = 0; i < input.size(); i++) {
    if (i % 28 == 0) std::cout << '\n';
    if (input[i] < 0.33 && input[i] >= 0) std::cout << '.';
    else if (input[i] >= 0.33 && input[i] < 0.66) std::cout << '*';
    else if (input[i] >= 0.66 && input[i] <= 1.0) std::cout << '#';
    else std::cout << "CORRUPT!";
  }
}

template <typename T>
void print_output(const std::vector<Value<T>>& output) {
  std::cout << "Output(";
  auto it = output.begin();
  while (it < output.end() - 1) {
    std::cout << *it << ", ";
    ++it;
  }
  std::cout << *it << ")\n";
}

inline void print_target(const std::vector<uint8_t>& target) {
  std::cout << "Target(";
  auto it = target.begin();
  while (it < target.end() - 1) {
    std::cout << static_cast<int>(*it) << ", ";
    ++it;
  }
  std::cout << static_cast<int>(*it) << ")\n";
}

inline void print_target(const uint8_t target) {
  std::cout << "Target(";
  constexpr size_t len = 10;
  std::vector<uint8_t> ohe(len, 0);
  ohe[target] = 1;
  print_target(ohe);
}

template<typename T, class Loss, class Input_Tp, class Target_Tp>
void train_model(MLP<T>& model,
                 const Input_Tp& inputs,
                 const Target_Tp& targets,
                 Loss& loss,
                 const double learning_rate,
                 const size_t epochs) {
  const auto num_samples = inputs.size();
  for (size_t e = 0; e < epochs; e++) {
    double epoch_loss = 0;
    for (size_t i = 0; i < num_samples; i++) {
      loss.compute_loss(inputs, targets);
      epoch_loss += loss.get();
      loss.backward();
      model.step(learning_rate);
      model.zero_grad();
      loss.zero();
    }
    std::cout << "Epoch " << e << ": "
              << "Loss = " << epoch_loss / num_samples
              << '\n';
  }
}

#endif //UTILS_HPP
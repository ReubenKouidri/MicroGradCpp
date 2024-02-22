#ifndef UTILS_HPP
#define UTILS_HPP

#include <iostream>
#include <vector>

template <typename T>
class MLP;

inline void visualise_input(const std::vector<double> &input) {
  for (std::size_t i = 0; i < input.size(); i++) {
    if (i%28==0) std::cout << '\n';
    if (input[i] < 0.33 && input[i] >= 0) std::cout << '.';
    else if (input[i] >= 0.33 && input[i] < 0.66) std::cout << '*';
    else if (input[i] >= 0.66 && input[i] <= 1.0) std::cout << '#';
    else std::cout << "CORRUPT!";
  }
}

template <typename T>
void print_output(const std::vector<Value<T>> &output) {
  std::cout << "Output(";
  auto it = output.begin();
  while (it < output.end() - 1) {
    std::cout << *it << ", ";
    ++it;
  }
  std::cout << *it << ")\n";
}

inline void print_target(const std::vector<uint8_t> &target) {
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
  constexpr std::size_t len = 10;
  std::vector<uint8_t> ohe(len, 0);
  ohe[target] = 1;
  print_target(ohe);
}

#endif //UTILS_HPP
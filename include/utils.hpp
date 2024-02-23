#ifndef UTILS_HPP
#define UTILS_HPP

#include <iostream>
#include <vector>
#include <ranges>

template <typename T>
class MLP;

template <typename T>
[[maybe_unused]] void visualise_input(const std::vector<T>& input) {
  static constexpr T t1{0.33};
  static constexpr T t2{0.67};
  for (std::size_t i = 0; i < input.size(); i++) {
    if (i % 28 == 0) std::cout << '\n';
    if (input[i] < t1 && input[i] >= static_cast<T>(0)) std::cout << '.';
    else if (input[i] >= t1 && input[i] < t2) std::cout << '*';
    else if (input[i] >= t2 && input[i] <= static_cast<T>(1)) std::cout << '#';
    else std::cout << "CORRUPT!";
  }
}

template <typename T>
[[maybe_unused]] void print_output(const std::vector<Value<T>>& output) {
  std::cout << "Output(";
  for (const auto& val : output | std::views::take(output.size() - 1)) {
    std::cout << val << ", ";
  }
  if (!output.empty()) std::cout << output.back() << ")\n";
}

[[maybe_unused]] inline void print_target(const std::vector<uint8_t>& target) {
  std::cout << "Target(";
  for (const auto& t : target | std::views::take(target.size() - 1)) {
    std::cout << static_cast<int>(t) << ", ";
  }
  if (!target.empty()) std::cout << static_cast<int>(target.back()) << ")\n";
}

[[maybe_unused]] inline void print_target(const uint8_t target) {
  std::cout << "Target(";
  // should remove magic num, but this denotes the size of the target
  constexpr std::size_t len = 10;
  std::vector<uint8_t> ohe(len, 0);
  ohe[target] = 1;
  print_target(ohe);
}

#endif //UTILS_HPP
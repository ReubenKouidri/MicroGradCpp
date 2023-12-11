#include <iostream>
#include <random>
#include "src/module.hpp"
#include "src/losses.hpp"

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
  os << "Vector(";
  for (auto ptr=v.begin(); ptr!=v.end(); ++ptr)
  {
    os << *ptr;
    if (ptr != v.end()-1)
      os << ", ";
  }
  os << ")";
  return os;
}

int main() {
  srand(static_cast<unsigned int>(time(nullptr)));
  constexpr size_t epochs = 100;
  constexpr double learning_rate = 0.01;

  const MLP<double> model({5, 5, 5, 5});
  const std::vector<double>& input = {0.4, 0.5, 1.0, 0.3, 0.7};
  const std::vector<double> target = {1, -1, 1, -1, 1};

  auto loss = MSELoss(model, learning_rate);
  for (size_t e = 0; e < epochs; e++) {
    auto l = loss.compute_loss(input, target);
    l.backward();
    loss.step();
    loss.zero_grad();

  }

  return 0;
}
#include <iostream>
#include "src/module.hpp"

int main() {
  constexpr size_t nin = 3;
  const std::vector<size_t> neurons {4, 4, 2};
  const std::vector<double> inputs {1.0, 1.5, 2.0};
  const std::vector<double> targets {1.0, 0.0};
  auto mlp = MLP<double>(nin, neurons);
  auto mse_loss_ = MSELoss(mlp, targets);

  return 0;
}
#include <gtest/gtest.h>
#include "../src/neuron.hpp"
#include "../src/value.hpp"

class MLPTest : public ::testing::Test {
protected:
  void SetUp() override {
  }
  const size_t nin = 3;
  const std::vector<size_t> neurons {4, 4, 2};
  const std::vector<double> inputs {1.0, 1.5, 2.0};
  const std::vector<double> targets {1.0, 0.0};
  MLP<double> mlp_ = MLP<double>(nin, neurons);
  MSELoss<double> mse_loss_ = MSELoss(mlp_, targets);
};

TEST_F(MLPTest, forward) {
  mse_loss_.compute_loss(inputs);
}

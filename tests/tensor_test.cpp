#include <gtest/gtest.h>
#include "../src/tensor.hpp"

class TensorTest : public ::testing::Test {
protected:
  void SetUp() override {
  }

  Tensor<double> t0 = Tensor(0.0);
  Tensor<double> t1 = Tensor(1.0);
  Tensor<double> t2 = Tensor(2.0);
  Tensor<double> t3 = Tensor(3.0);
};

TEST_F(TensorTest, AddOperator) {
  auto t = t0 + t1;
  EXPECT_EQ(t0.get_data(), 0.0);
  EXPECT_EQ(t1.get_data(), 1.0);
  EXPECT_EQ(t.get_data(), 1.0);
}

TEST_F(TensorTest, NegationOperator) {
  -t1;
  EXPECT_EQ(t1.get_data(), -1.0);
}

TEST_F(TensorTest, SubtractionOperator) {
  auto t = t3 - t2;
  EXPECT_EQ(t.get_data(), 1.0);
  EXPECT_EQ(t.get_grad(), 0.0);
  // t3 and t2 should be unchanged
  EXPECT_EQ(t3.get_data(), 3.0);
  EXPECT_EQ(t3.get_grad(), 0.0);
  EXPECT_EQ(t2.get_data(), 2.0);
  EXPECT_EQ(t2.get_grad(), 0.0);
}

TEST_F(TensorTest, GradientRegistrationAdd) {
  auto add = t1 + t0;
  add.backward();
  EXPECT_EQ(add.get_data(), 1.0);
  EXPECT_EQ(add.get_grad(), 1.0);
  EXPECT_EQ(t0.get_data(), 0.0);
  EXPECT_EQ(t0.get_grad(), 1.0);
  EXPECT_EQ(t1.get_data(), 1.0);
  EXPECT_EQ(t1.get_grad(), 1.0);
}

TEST_F(TensorTest, GradientRegistrationSub) {
  auto subtract = t3 - t2;
  subtract.backward();
  EXPECT_EQ(subtract.get_data(), 1.0);
  EXPECT_EQ(subtract.get_grad(), 1.0);
  EXPECT_EQ(t2.get_data(), 2.0);
  EXPECT_EQ(t2.get_grad(), -1.0);
  EXPECT_EQ(t3.get_data(), 3.0);
  EXPECT_EQ(t3.get_grad(), 1.0);
}

TEST_F(TensorTest, GradientRegistrationMul) {
  auto multiply = t2 * t3;
  multiply.backward();
  EXPECT_EQ(multiply.get_data(), 6.0);
  EXPECT_EQ(multiply.get_grad(), 1.0);
  EXPECT_EQ(t2.get_data(), 2.0);
  EXPECT_EQ(t2.get_grad(), 3.0);
  EXPECT_EQ(t3.get_data(), 3.0);
  EXPECT_EQ(t3.get_grad(), 2.0);
}

TEST_F(TensorTest, GradientRegistrationDiv) {
  auto divide = t1 / t2;
  divide.backward();
  EXPECT_EQ(divide.get_data(), 0.5);
  EXPECT_EQ(divide.get_grad(), 1.0);
  EXPECT_EQ(t1.get_data(), 1.0);
  EXPECT_EQ(t1.get_grad(), 0.5);
  EXPECT_EQ(t2.get_data(), 2.0);
  EXPECT_EQ(t2.get_grad(), -0.25);
}

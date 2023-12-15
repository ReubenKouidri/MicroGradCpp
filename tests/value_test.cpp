#include <gtest/gtest.h>
#include "../include/value.hpp"

class ValueTest : public ::testing::Test {
protected:
  void SetUp() override {
  }

  Value<double> t0 = Value(0.0);
  Value<double> t1 = Value(1.0);
  Value<double> t2 = Value(2.0);
  Value<double> t3 = Value(3.0);
};

TEST_F(ValueTest, AddOperator) {
  auto t = t0 + t1;
  EXPECT_EQ(t0.get_data(), 0.0);
  EXPECT_EQ(t1.get_data(), 1.0);
  EXPECT_EQ(t.get_data(), 1.0);
}

TEST_F(ValueTest, NegationOperator) {
  t1 = -t1;
  EXPECT_EQ(t1.get_data(), -1.0);
}

TEST_F(ValueTest, SubtractionOperator) {
  auto t = t3 - t2;
  EXPECT_EQ(t.get_data(), 1.0);
  EXPECT_EQ(t.get_grad(), 0.0);
  // t3 and t2 should be unchanged
  EXPECT_EQ(t3.get_data(), 3.0);
  EXPECT_EQ(t3.get_grad(), 0.0);
  EXPECT_EQ(t2.get_data(), 2.0);
  EXPECT_EQ(t2.get_grad(), 0.0);
}

TEST_F(ValueTest, GradientRegistrationAdd) {
  auto add = t1 + t0;
  add.backward();
  EXPECT_EQ(add.get_data(), 1.0);
  EXPECT_EQ(add.get_grad(), 1.0);
  EXPECT_EQ(t0.get_data(), 0.0);
  EXPECT_EQ(t0.get_grad(), 1.0);
  EXPECT_EQ(t1.get_data(), 1.0);
  EXPECT_EQ(t1.get_grad(), 1.0);
}

TEST_F(ValueTest, GradientRegistrationSub) {
  auto subtract = t3 - t2;
  subtract.backward();
  EXPECT_EQ(subtract.get_data(), 1.0);
  EXPECT_EQ(subtract.get_grad(), 1.0);
  EXPECT_EQ(t2.get_data(), 2.0);
  EXPECT_EQ(t2.get_grad(), -1.0);
  EXPECT_EQ(t3.get_data(), 3.0);
  EXPECT_EQ(t3.get_grad(), 1.0);
}

TEST_F(ValueTest, GradientRegistrationMul) {
  auto multiply = t2 * t3;
  multiply.backward();
  EXPECT_EQ(multiply.get_data(), 6.0);
  EXPECT_EQ(multiply.get_grad(), 1.0);
  EXPECT_EQ(t2.get_data(), 2.0);
  EXPECT_EQ(t2.get_grad(), 3.0);
  EXPECT_EQ(t3.get_data(), 3.0);
  EXPECT_EQ(t3.get_grad(), 2.0);
}

TEST_F(ValueTest, GradientRegistrationDiv) {
  auto divide = t1 / t2;
  divide.backward();
  EXPECT_EQ(divide.get_data(), 0.5);
  EXPECT_EQ(divide.get_grad(), 1.0);
  EXPECT_EQ(t1.get_data(), 1.0);
  EXPECT_EQ(t1.get_grad(), 0.5);
  EXPECT_EQ(t2.get_data(), 2.0);
  EXPECT_EQ(t2.get_grad(), -0.25);
}

TEST_F(ValueTest, ActivationRelu) {
  const auto t5 = t2 + t3;
  auto a = relu(t5);
  EXPECT_EQ(a.get_data(), 5.0);
  EXPECT_EQ(t2.get_data(), 2.0);
  EXPECT_EQ(t3.get_data(), 3.0);
  // this = t5, out = a
  a.backward();
  EXPECT_EQ(a.get_grad(), 1.0);
  EXPECT_EQ(a.get_data(), 5.0);
  // (1 - 5^2) * 1
  EXPECT_EQ(t5.get_grad(), 1.0);
  EXPECT_EQ(t5.get_data(), 5.0);
}

TEST_F(ValueTest, ActivationTanh) {
  const auto t5 = t2 + t3;
  auto a = t5.activation_output(Activation::TANH);
  EXPECT_NEAR(a.get_data(), 1.0, 0.001);
  EXPECT_EQ(t2.get_data(), 2.0);
  EXPECT_EQ(t3.get_data(), 3.0);
  // this = t5, out = a
  a.backward();
  EXPECT_EQ(a.get_grad(), 1.0);
  EXPECT_NEAR(a.get_data(), 1.0, 0.001);
  // (1 - a^2) * 1
  EXPECT_NEAR(t5.get_grad(), 0.0, 0.001);
  EXPECT_EQ(t5.get_data(), 5.0);
  EXPECT_EQ(t2.get_data(), 2.0);
  EXPECT_EQ(t3.get_data(), 3.0);
}

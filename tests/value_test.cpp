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

  Value<double> s0 = Value(0.0);
  Value<double> s1 = Value(0.1);
  Value<double> s2 = Value(0.2);
  Value<double> s3 = Value(0.3);

  Value<double> m1 = Value(-1.0);
};

TEST_F(ValueTest, AddOperator) {
  auto t = t0 + t1;
  EXPECT_EQ(t0.get_data(), 0.0);
  EXPECT_EQ(t1.get_data(), 1.0);
  EXPECT_EQ(t.get_data(), 1.0);
  t.backward();
  EXPECT_EQ(t.get_data(), 1.0);
  EXPECT_EQ(t0.get_data(), 0.0);
  EXPECT_EQ(t1.get_data(), 1.0);
  EXPECT_EQ(t.get_grad(), 1.0);
  EXPECT_EQ(t0.get_grad(), 1.0);
  EXPECT_EQ(t1.get_grad(), 1.0);
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
  auto t = t2*t3;
  EXPECT_EQ(t.get_data(), 6.0);
  EXPECT_EQ(t2.get_data(), 2);
  EXPECT_EQ(t3.get_data(), 3);

  EXPECT_EQ(t.get_grad(), 0.0);
  EXPECT_EQ(t2.get_grad(), 0.0);
  EXPECT_EQ(t3.get_grad(), 0.0);

  t.backward();
  /* IMPORTANT!!
   * If gradients are being clipped to 1.0 then this is correct! */

  EXPECT_EQ(t.get_data(), 6.0);
  EXPECT_EQ(t.get_grad(), 1.0);

  EXPECT_EQ(t2.get_data(), 2.0);
  EXPECT_EQ(t2.get_grad(), 1.0);

  EXPECT_EQ(t3.get_data(), 3.0);
  EXPECT_EQ(t3.get_grad(), 1.0);

  auto s = s2*s3;
  s.backward();
  EXPECT_EQ(s.get_data(), 0.06);
  EXPECT_EQ(s2.get_data(), 0.2);
  EXPECT_EQ(s2.get_grad(), 0.3);
  EXPECT_EQ(s3.get_data(), 0.3);
  EXPECT_EQ(s3.get_grad(), 0.2);

}

TEST_F(ValueTest, GradientRegistrationDiv) {
  auto divide = t1/t2;
  divide.backward();
  EXPECT_EQ(divide.get_data(), 0.5);
  EXPECT_EQ(divide.get_grad(), 1.0);
  EXPECT_EQ(t1.get_data(), 1.0);
  EXPECT_EQ(t1.get_grad(), 0.5);
  EXPECT_EQ(t2.get_data(), 2.0);
  EXPECT_EQ(t2.get_grad(), -0.25);
}

TEST_F(ValueTest, Relu) {
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

TEST_F(ValueTest, Pow) {
  auto t = ops::pow(s2, 2);
  EXPECT_NEAR(t.get_data(), 0.04, 1e-7);
  EXPECT_EQ(s2.get_data(), 0.2);
  t.backward();
  EXPECT_EQ(t.get_grad(), 1.0);
  EXPECT_EQ(s2.get_data(), 0.2);
  EXPECT_NEAR(s2.get_grad(), 0.4, 1e-7);
}

TEST_F(ValueTest, Exp) {
  auto t = ops::exp(t2);
  EXPECT_NEAR(t.get_data(), 7.38905609893065, 1e-7);
  EXPECT_EQ(t2.get_data(), 2.0);
  t.backward();
  EXPECT_EQ(t.get_grad(), 1.0);
  EXPECT_EQ(t2.get_data(), 2.0);
  /* CLIPPING GRADS */
  EXPECT_NEAR(t2.get_grad(), 1.0, 1e-7);

  auto s = ops::exp(m1);
  s.backward();
  EXPECT_NEAR(s.get_data(), 0.36787944117, 1e-7);
  EXPECT_NEAR(m1.get_grad(), 0.36787944117, 1e-7);

}

TEST_F(ValueTest, Log) {
  auto t4 = ops::log(t2);
  EXPECT_NEAR(t4.get_data(), 0.6931471806, 1e-7);
  EXPECT_EQ(t2.get_data(), 2.0);
  t4.backward();
  EXPECT_EQ(t4.get_grad(), 1.0);
  EXPECT_EQ(t2.get_data(), 2.0);
  EXPECT_EQ(t2.get_grad(), 0.5);
}
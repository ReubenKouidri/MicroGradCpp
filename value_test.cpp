#include <gtest/gtest.h>
#include "value.hpp"

class ValueTest : public ::testing::Test {
protected:
  void SetUp() override {
  }

  Value v0_ {0.5};
  Value v1_ {1.0};
};


TEST_F(ValueTest, AddOperator) {
  // operators
  // data
  auto v = v0_ + v1_;
  EXPECT_EQ(v0_.get_data(), 0.5);
  EXPECT_EQ(v1_.get_data(), 1.0);
  EXPECT_EQ(v.get_data(), 1.5);
  // BACKWARD
  v.backward();
  // DATA CHECK
  EXPECT_EQ(v0_.get_data(), 0.5);
  EXPECT_EQ(v1_.get_data(), 1.0);
  EXPECT_EQ(v.get_data(), 1.5);
  // GRAD CHECK
  EXPECT_EQ(v0_.get_grad(), 1.0);
  EXPECT_EQ(v1_.get_grad(), 1.0);
  EXPECT_EQ(v.get_grad(), 1.0);
  // PARENT CHECK
}

// Test for negation operator
TEST_F(ValueTest, NegationOperator) {
  -v0_;
  EXPECT_EQ(v0_.get_data(), -0.5);
}

// Test for multiplication operator
TEST_F(ValueTest, MultiplicationOperatorValueType) {
  auto v = v0_ * v1_;

  // Check if multiplication is performed correctly
  EXPECT_EQ(v0_.get_data(), 0.5);
  EXPECT_EQ(v1_.get_data(), 1.0);
  EXPECT_EQ(v.get_data(), 0.5);

  // Check gradients after backward propagation
  v.backward();
  EXPECT_EQ(v0_.get_grad(), 1.0);
  EXPECT_EQ(v1_.get_grad(), 0.5);
  EXPECT_EQ(v.get_grad(), 1.0);
}

TEST_F(ValueTest, ParentsCheck) {
  // Create some values
  const auto v = v0_ + v1_;
  // Get the parent pointers of v2_
  const auto& parents = v.get_parents();
  for (const auto& p : parents) {
    std::cout << p.get() << '\n';
  }

  // Check if v0_ and v1_ are present in v2_'s parents
  EXPECT_TRUE(std::any_of(parents.begin(), parents.end(), [&](const auto& ptr) {
      return ptr.get() == &v0_ || ptr.get() == &v1_;
  }));
}

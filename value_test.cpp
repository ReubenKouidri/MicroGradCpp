#include <gtest/gtest.h>
#include "value.hpp"


class ValueTest : public ::testing::Test {
protected:
  void SetUp() override {

  }

  Value v0_ {0.0};
  Value v1_ {1.0};
  Value v2_ {v0_ + v1_};

};


TEST_F(ValueTest, AddOperator) {
  // operators
  EXPECT_EQ(v0_.op, "");
  EXPECT_EQ(v1_.op, "");
  EXPECT_EQ(v2_.op, "+");

  // data
  EXPECT_EQ(v0_.data, 0.0);
  EXPECT_EQ(v1_.data, 1.0);
  EXPECT_EQ(v2_.data, 1.0);
}
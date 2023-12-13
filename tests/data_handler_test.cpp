#include <gtest/gtest.h>
#include <fstream>
#include "../include/data_handler.hpp"

class DataHandlerTest : public ::testing::Test {
protected:
  void SetUp() override {
  }
  const std::string image_file = "../data/train-images-idx3-ubyte";
  const std::string label_file = "../data/train-labels-idx1-ubyte";
  DataHandler dh = DataHandler();
};

TEST_F(DataHandlerTest, read_header) {
  std::array<uint32_t, 4> header{};
  std::ifstream bytes_file(image_file, std::ios::binary);
  DataHandler::read_header(header, bytes_file);
  ASSERT_EQ(header[0], 2051);
  ASSERT_EQ(header[1], 60000);
  ASSERT_EQ(header[2], 28);
  ASSERT_EQ(header[3], 28);
  bytes_file.close();

  std::array<uint32_t, 2> header2 {};
  bytes_file.open(label_file, std::ios::binary);
  DataHandler::read_header(header2, bytes_file);
  ASSERT_EQ(header2[0], 2049);
  ASSERT_EQ(header2[1], 60000);
  bytes_file.close();
}

TEST_F(DataHandlerTest, read_features) {
  dh.read_feature_vector(image_file);
  ASSERT_EQ(dh.get_image_size(), 784);
  ASSERT_EQ(dh.get_all_data()->size(), 60000);
  for (const auto* d : *dh.get_all_data()) {
    ASSERT_EQ(d->get_label(), 0);
  }
  dh.read_feature_labels(label_file);
}

TEST_F(DataHandlerTest, split_data) {
  dh.read_feature_vector(image_file);
  dh.read_feature_labels(label_file);
  dh.split_data();
  ASSERT_EQ(dh.get_training_data()->size(), 48000);
  ASSERT_EQ(dh.get_validation_data()->size(), 6000);
  ASSERT_EQ(dh.get_test_data()->size(), 6000);
}

TEST_F(DataHandlerTest, count_classes) {
  dh.read_feature_vector(image_file);
  dh.read_feature_labels(label_file);
  ASSERT_EQ(dh.num_classes(), 10);
  dh.print_class_info();
}

TEST_F(DataHandlerTest, normalise_data) {
  dh.read_feature_vector(image_file);
  dh.read_feature_labels(label_file);
  dh.normalise_data();
  dh.split_data();
  const auto data = dh.get_training_data()->at(0);
  for (const auto v : *data->get_feature_vector()) {
    ASSERT_GE(v, 0.0);
    ASSERT_LE(v, 1.0);
  }
}


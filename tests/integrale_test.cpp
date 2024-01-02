#include <string>
#include <tuple>
#include <gtest/gtest.h>
#include "../include/data_handler.hpp"
#include "../include/components.hpp"
#include "../include/losses.hpp"

class IntegralTest : public testing::Test {
 public:
  IntegralTest() : image_file("../data/t10k-images-idx3-ubyte"),
                   label_file("../data/t10k-labels-idx1-ubyte"),
                   data_handler(image_file, label_file),
                   image_size(data_handler.get_image_size()),
                   num_classes(data_handler.num_classes()),
                   model({
                             Layer<double>{image_size, 128, UnaryOp::relu},
                             Layer<double>{128, num_classes,
                                           UnaryOp::softmax}}),
                   batched_data(data_handler.get_batched_test_data(10)) {
    std::tie(inputs, targets) = extract(batched_data);
    if (!inputs.empty() && !inputs[0].empty()) {
      single_image_batch = inputs[0];
    }
    if (!targets.empty() && !targets[0].empty()) {
      single_target_batch = targets[0];
    }
  }

 protected:
  void SetUp() override {}

  const std::string image_file;
  const std::string label_file;
  const DataHandler data_handler;
  const size_t image_size;
  const size_t num_classes;
  MLP<double> model;

  std::vector<std::vector<Data *>> batched_data;
  std::vector<std::vector<image_t>> inputs;  // batched
  std::vector<std::vector<label_t>> targets;  // batched
  std::vector<image_t> single_image_batch;
  std::vector<label_t> single_target_batch;

  static constexpr double learning_rate{0.01};
  static constexpr size_t epochs{2};
  static constexpr size_t batch_size = 100;
};

TEST_F(IntegralTest, test_SCCE_single) {
  std::cout << "========================================\n";
  visualise_input(single_image_batch[0]);
  std::cout << static_cast<int>(single_target_batch[0]) << '\n';
  SparseCCELoss<double> sparse_cce_loss(&model);
  train_model(model,
              single_image_batch,
              single_target_batch,
              sparse_cce_loss,
              learning_rate,
              epochs);
};

// TEST_F(IntegralTest, test_SCCE_batch) {
//   std::cout << "========================================\n";
//   visualise_input(single_image_batch[0]);
//   std::cout << " = " << static_cast<int>(single_target_batch[0]) << '\n';
//   const auto image_batch = std::vector{single_image_batch,
//                                        single_image_batch};
//   const auto target_batch = std::vector{single_target_batch,
//                                         single_target_batch};
//   SparseCCELoss<double> sparse_cce_loss(model, learning_rate);
//   train_model(model,
//               image_batch,
//               target_batch,
//               sparse_cce_loss,
//               learning_rate,
//               epochs);
// };
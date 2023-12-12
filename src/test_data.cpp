#include <string>
#include "../include/data_handler.hpp"
#include "../include/module.hpp"
#include "../include/losses.hpp"


int main() {
  const std::string image_file = "../data/train-images-idx3-ubyte";
  const std::string label_file = "../data/train-labels-idx1-ubyte";
  DataHandler data_handler;

  // Read the feature vectors and labels
  data_handler.read_feature_vector(image_file);
  data_handler.read_feature_labels(label_file);

  // Display some information
  data_handler.count_classes();

  // Access the training, validation, and test data
  const std::vector<Data*>* training_data = data_handler.get_training_data();
  const std::vector<Data*>* validation_data = data_handler.get_validation_data();
  const std::vector<Data*>* test_data = data_handler.get_test_data();

  constexpr size_t epochs = 10;
  constexpr double learning_rate = 0.01;
  const size_t image_size = data_handler.image_size();
  size_t num_classes = data_handler.num_classes();
  const MLP<double> model({image_size, 128, num_classes});

  auto loss = MSELoss(model, learning_rate);
  for (size_t e = 0; e < epochs; e++) {
    for (const auto d: *training_data) {
      auto target = d->get_label();
      auto input = *(d->get_feature_vector());
      auto l = loss.compute_loss(input, target);
      l.backward();
      loss.step();
      loss.zero_grad();
    }
  }
  return 0;
}

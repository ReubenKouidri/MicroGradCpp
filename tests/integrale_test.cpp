#include <string>
#include "../include/data_handler.hpp"
#include "../include/module.hpp"
#include "../include/losses.hpp"

template<typename T>
Value<T> train_step(SparseCCELoss<T>& loss, const std::vector<Data*>& batch) {

  std::vector<uint8_t> batched_targets;
  std::vector<std::vector<T>> batched_inputs;

  batched_inputs.reserve(batch.size());
  batched_targets.reserve(batch.size());

  for (const auto* data : batch) {
    if (data) {
      batched_targets.push_back(data->get_label());
      batched_inputs.push_back(*data->get_feature_vector());
    }
  }

  auto l = loss.compute_loss(batched_inputs, batched_targets);
  l.backward();
  loss.step();
  loss.zero_grad();

  return l;
}

int main() {
  const std::string image_file = "../data/t10k-images-idx3-ubyte";
  const std::string label_file = "../data/t10k-labels-idx1-ubyte";
  const DataHandler data_handler(image_file, label_file);

  constexpr size_t batch_size = 100;
  // Access the training, validation, and test data
  const auto batched_training_data = data_handler.get_batched_training_data(batch_size);

  constexpr size_t epochs = 10;
  constexpr double learning_rate = 0.01;

  const size_t image_size = data_handler.get_image_size();
  const size_t num_classes = data_handler.num_classes();

  const Layer<double> layer0(image_size, 128, Activation::RELU);
  const Layer<double> layer1(128, num_classes, Activation::SOFTMAX);
  const MLP<double> model({layer0, layer1});
  auto loss = SparseCCELoss<double>(model, learning_rate);

  for (size_t e = 0; e < epochs; e++) {
    std::cout << "Start of epoch " << e << '\n';
    for (const auto& batch: batched_training_data) {
      auto loss_val = train_step(loss, batch);
      std::cout << loss_val << '\n';
    }
  }
  return 0;
}

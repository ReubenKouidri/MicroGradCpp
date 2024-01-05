#include <string>
#include <vector>
#include <tuple>
#include <iostream>
#include "include/components.hpp"
#include "include/data_handler.hpp"
#include "include/losses.hpp"
#include "include/optimiser.hpp"
#include "include/trainer.hpp"

constexpr size_t batch_size = 100;
constexpr size_t epochs = 1;
constexpr double learning_rate = 1e-3;

int main() {

  const std::string image_file{"data/t10k-images-idx3-ubyte"};
  const std::string label_file{"data/t10k-labels-idx1-ubyte"};

  const auto dh{DataHandler(image_file, label_file)};
  const size_t image_size = dh.get_image_size();
  const size_t num_classes = dh.num_classes();
  const auto batched_training_data{dh.get_batched_training_data(batch_size)};
  const auto validation_data{dh.get_validation_data()};

  std::vector<std::vector<image_t>> batched_training_images;
  std::vector<std::vector<label_t>> batched_training_targets;
  std::vector<image_t> validation_images;
  std::vector<label_t> validation_targets;

  std::tie(batched_training_images,
           batched_training_targets) = extract(batched_training_data);
  std::tie(validation_images,
           validation_targets) = extract(validation_data);

  const MLP<double> model({
                              Layer<double>{image_size, 32, UnaryOp::relu},
                              Layer<double>{32, num_classes, UnaryOp::softmax}
                          });
  const std::shared_ptr<const MLP<double>>
      mp = std::make_shared<MLP<double>>(model);
  auto adam{Adam<double>(mp, learning_rate)};
  auto loss{SparseCCELoss<double>(mp)};

  train_batched_dataset(mp,
                        batched_training_images,
                        batched_training_targets,
                        validation_images,
                        validation_targets,
                        loss,
                        adam,
                        epochs);

  std::cout << "Done\n";

  return 0;
}
#include <string>
#include <vector>
#include <tuple>
#include <iostream>
#include "include/components.hpp"
#include "include/data_handler.hpp"
#include "include/losses.hpp"
#include "include/optimiser.hpp"

template <typename T, class Loss>
inline void train_batched(const MLP<T> *const model,
                          const std::vector<typename Loss::batched_input_type> &inputs,
                          const std::vector<typename Loss::batched_target_type> &targets,
                          Loss &loss,
                          Adam<T> &optimiser,
                          const size_t epochs) {
  const auto num_batches = inputs.size();
  for (size_t e = 0; e < epochs; e++) {
    double epoch_loss = 0;

    for (size_t i = 0; i < num_batches; i++) {
      auto start = std::chrono::high_resolution_clock::now();
      loss.compute_loss(inputs[i], targets[i]);
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> duration{end - start};
      std::cout << "batch " << i+1 << "/" << num_batches << " forward: " << duration.count() << " seconds\n";
      epoch_loss += loss.get();

      start = std::chrono::high_resolution_clock::now();
      loss.backward();
      end = std::chrono::high_resolution_clock::now();
      duration = end - start;
      std::cout << "backward: " << duration.count() << " seconds\n";

      optimiser.step();
      model->zero_grad();
      loss.zero();
    }
    std::cout << "Epoch " << e << ": "
              << "Loss = " << epoch_loss/num_batches
              << '\n';
  }
}

int main() {
  // test loop
  // print results

  const std::string image_file{"data/t10k-images-idx3-ubyte"};
  const std::string label_file{"data/t10k-labels-idx1-ubyte"};
  const auto dh{DataHandler(image_file, label_file)};
  const size_t image_size = dh.get_image_size();
  const size_t num_classes = dh.num_classes();
  constexpr size_t batch_size = 50;
  constexpr size_t epochs = 2;
  auto batched_data{dh.get_batched_test_data(batch_size)};
  std::vector<std::vector<image_t>> batched_images;
  std::vector<std::vector<label_t>> batched_targets;
  std::tie(batched_images, batched_targets) = extract(batched_data);

  const MLP<double> model({
                              Layer<double>{image_size, 128, UnaryOp::relu},
                              Layer<double>{128, num_classes, UnaryOp::softmax}
                          });
  const MLP<double> *const mp = &model;
  auto adam{Adam<double>(mp, 0.01)};
  auto loss{SparseCCELoss<double>(mp)};

  std::cout << "Running...\n";
  train_batched(mp, batched_images, batched_targets, loss, adam, epochs);
  std::cout << "Done\n";

  return 0;
}
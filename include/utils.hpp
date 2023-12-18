#ifndef UTILS_HPP
#define UTILS_HPP

#include <iostream>
#include <vector>
#include "data.hpp"
#include "module.hpp"

using image_t = std::vector<double>;
using label_t = uint8_t;
using data_vec_t = std::vector<Data*>;
using data_batch_t = std::vector<data_vec_t>;

inline void visualise_input(const image_t& input) {
  for (size_t i = 0; i < input.size(); i++) {
    if (i % 28 == 0) std::cout << '\n';
    if (input[i] < 0.33 && input[i] >= 0) std::cout << '.';
    else if (input[i] >= 0.33 && input[i] < 0.66) std::cout << '*';
    else if (input[i] >= 0.66 && input[i] <= 1.0) std::cout << '#';
    else std::cout << "CORRUPT!";
  }
}

template<typename T, class Loss, class Input_Tp, class Target_Tp>
void train_model(MLP<T>& model,
                 const Input_Tp& inputs,
                 const Target_Tp& targets,
                 Loss& loss,
                 const double learning_rate,
                 const size_t epochs) {
  const auto num_samples = inputs.size();
  for (auto e = 0; e < epochs; e++) {
    double epoch_loss = 0;
    for (auto i = 0; i < num_samples; i++) {
      auto grads = loss.compute_gradients(inputs[i], targets[i]);
      epoch_loss += loss.get();
      model.backward(grads);
      model.step(learning_rate);
      model.zero_grad();
      loss.zero();
    }
    std::cout << "Epoch " << e << ": "
              << "Loss = " << epoch_loss / num_samples
              << '\n';
  }
}

std::tuple<image_t, label_t>
inline extract_single(const Data* d) {
  return std::make_tuple(*d->get_feature_vector(), d->get_label());
}

std::tuple<std::vector<image_t>, std::vector<label_t>>
inline extract(const data_vec_t* data) {
  std::vector<image_t> inputs;
  std::vector<label_t> targets;
  inputs.reserve(data->size());
  targets.reserve(data->size());

  for (const auto d : *data) {
    if (d) {
      auto [img, lbl] = extract_single(d);
      inputs.push_back(std::move(img));
      targets.push_back(lbl);
    }
  }
  return std::make_tuple(std::move(inputs), std::move(targets));
}

std::tuple<std::vector<std::vector<image_t>>, std::vector<std::vector<label_t>>>
inline extract(const data_batch_t& batched_data) {
  std::vector<std::vector<image_t>> img_batch;
  std::vector<std::vector<label_t>> lbl_batch;
  img_batch.reserve(batched_data.size());
  lbl_batch.reserve(batched_data.size());

  for (const auto& data_vec : batched_data) {
    auto [imgs, lbls] = extract(&data_vec);
    img_batch.push_back(std::move(imgs));
    lbl_batch.push_back(std::move(lbls));
  }
  return std::make_tuple(std::move(img_batch), std::move(lbl_batch));
}

#endif //UTILS_HPP
#ifndef UTILS_HPP
#define UTILS_HPP

#include <iostream>
#include <vector>
#include "data.hpp"

template <typename T>
class MLP;

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

template<typename T, typename... Args>
T generate_weight(const UnaryOp& activation, Args... args) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<T> dist;

  if (activation == UnaryOp::relu) {
    /* Calculate He initialization using the first argument */
    auto nin = std::get<0>(std::make_tuple(args...));
    return std::normal_distribution<T>(
      0, std::sqrt(2.0 / static_cast<double>(nin)))(gen);
  }
  if (activation == UnaryOp::softmax) {
    /* Calculate Xavier initialization using both arguments */
    auto [nin, nout] = std::make_tuple(args...);
    return std::normal_distribution<T>(
      0, std::sqrt(2.0 / static_cast<double>(nin + nout)))(gen);
  }
  std::cout << "Need to implement init method for this activation function\n";
  return T{};
}

template <typename T>
void print_output(const std::vector<Value<T>>& output) {
  std::cout << "Output(";
  auto it = output.begin();
  while (it < output.end() - 1) {
    std::cout << *it << ", ";
    ++it;
  }
  std::cout << *it << ")\n";
}

template<typename T, class Loss, class Input_Tp, class Target_Tp>
void train_model(MLP<T>& model,
                 const Input_Tp& inputs,
                 const Target_Tp& targets,
                 Loss& loss,
                 const double learning_rate,
                 const size_t epochs) {
  const auto num_samples = inputs.size();
  for (size_t e = 0; e < epochs; e++) {
    double epoch_loss = 0;
    for (size_t i = 0; i < num_samples; i++) {
      loss.compute_loss(inputs, targets);
      epoch_loss += loss.get();
      loss.backward();
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
inline extract(const Data* d) {
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
      auto [img, lbl] = extract(d);
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
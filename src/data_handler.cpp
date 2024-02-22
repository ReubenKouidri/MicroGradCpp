#include <iostream>
#include <fstream>
#include <unordered_set>
#include <random>
#include "../include/data_handler.hpp"

DataHandler::DataHandler() {
  data_array_ = new std::vector<Data *>;
  training_data_ = new std::vector<Data *>;
  validation_data_ = new std::vector<Data *>;
  test_data_ = new std::vector<Data *>;
}

[[maybe_unused]] DataHandler::DataHandler(const std::string &image_path,
                         const std::string &label_path)
    : DataHandler() {
  read_feature_vector(image_path);
  read_feature_labels(label_path);
  normalise_data();
  split_data();
  count_classes();
}

DataHandler::~DataHandler() {
  delete data_array_;
  delete training_data_;
  delete validation_data_;
  delete test_data_;
  data_array_ = nullptr;
  training_data_ = nullptr;
  validation_data_ = nullptr;
  test_data_ = nullptr;
}

void DataHandler::read_feature_vector(const std::string &path) {
  std::array<uint32_t, 4> header{}; // MAGIC|NUM_IMAGES|ROW_SIZE|COL_SIZE
  std::ifstream bytes_file(path, std::ios::binary);
  read_header(header, bytes_file);

  const std::size_t image_size = header[2]*header[3];
  image_size_ = image_size;

  while (bytes_file) {
    auto *d = new Data(image_size);
    std::vector<unsigned char> arr(image_size);
    if (!bytes_file.read(reinterpret_cast<char *>(arr.data()),
                         static_cast<long>(image_size))) {
      delete d;
      break;
    }
    d->get_feature_vector()->assign(arr.begin(), arr.end());
    data_array_->emplace_back(d);
  }
  total_length_ = header[1];
  std::cout << "Done extracting " << header[1] << " images\n";
}

template <std::size_t S>
void DataHandler::read_header(std::array<uint32_t, S> &header,
                              std::ifstream &bytes_file) {
  if (!bytes_file)
    throw std::runtime_error("Failed to open file.\n");

  for (std::size_t i = 0; i < header.size(); ++i) {
    if (!bytes_file.read(reinterpret_cast<char *>(&header[i]),
                         sizeof(uint32_t)))
      throw std::runtime_error(
          "Failed to read the header at index " + std::to_string(i) + ".\n"
      );
    header[i] = ntohl(header[i]); // convert big->little endian
  }
}

void DataHandler::read_feature_labels(const std::string &path) {
  std::array<uint32_t, 2> header{}; // MAGIC|NUM_LABELS
  std::ifstream bytes_file(path, std::ios::binary);
  read_header(header, bytes_file);

  for (std::size_t i = 0; i < header[1]; ++i) {
    uint8_t label;
    if (!bytes_file.read(reinterpret_cast<char *>(&label), sizeof(uint8_t)))
      throw std::runtime_error(
          "Failed to read the header at index " + std::to_string(i) + ".\n"
      );
    data_array_->at(i)->set_label(label);
  }
  std::cout << "Done extracting labels\n";
  count_classes();
}

void DataHandler::append_data(const std::vector<std::size_t> &indices,
                              std::vector<Data *> *dataset,
                              std::vector<Data *> *subset) {
  for (const auto index : indices) {
    subset->emplace_back(dataset->at(index));
  }
}

void DataHandler::split_data() const {
  const std::size_t train_size = data_array_->size()*TRAIN_SPLIT;
  const std::size_t validation_size = data_array_->size()*VALIDATION_SPLIT;

  std::vector<std::size_t> indices(total_length_);
  std::iota(indices.begin(), indices.end(), 0);

  std::random_device rd;
  std::mt19937 g(rd());
  std::ranges::shuffle(indices, g);

  const std::vector<std::size_t> train_indices(
      indices.begin(),
      indices.begin() + static_cast<long>(train_size));
  const std::vector<std::size_t> validation_indices(
      indices.begin() + static_cast<long>(train_size),
      indices.begin() + static_cast<long>(train_size + validation_size));
  const std::vector<std::size_t> test_indices(
      indices.begin() + static_cast<long>(train_size + validation_size),
      indices.end());

  append_data(train_indices, data_array_, training_data_);
  append_data(validation_indices, data_array_, validation_data_);
  append_data(test_indices, data_array_, test_data_);

  std::cout << "Training size: " << training_data_->size() << '\n';
  std::cout << "Validation size: " << validation_data_->size() << '\n';
  std::cout << "Test size: " << test_data_->size() << '\n';
}

void DataHandler::count_classes() {
  for (const Data *data : *data_array_) {
    const uint8_t label = data->get_label();
    if (!class_map_.contains(label)) {
      class_map_[label] = 1;
    }
    class_map_[label]++;
  }
}

std::size_t DataHandler::get_image_size() const { return image_size_; }
std::size_t DataHandler::num_classes() const { return class_map_.size(); }

const std::vector<Data *> *DataHandler::get_all_data() const {
  return data_array_;
}

const std::vector<Data *> *DataHandler::get_training_data() const {
  return training_data_;
}
const std::vector<Data *> *DataHandler::get_validation_data() const {
  return validation_data_;
}
const std::vector<Data *> *DataHandler::get_test_data() const {
  return test_data_;
}

std::vector<std::vector<Data *>> DataHandler::batch_dataset(
    const std::vector<Data *> *dataset,
    const std::size_t batch_size) {
  std::vector<std::vector<Data *>> batched_data;

  if (batch_size==0 || dataset->empty()) {
    throw std::runtime_error("Invalid batch size or empty dataset.");
  }

  const std::size_t num_batches = (dataset->size() + batch_size - 1)/batch_size;
  batched_data.reserve(num_batches);

  auto data_iter = dataset->cbegin();
  for (std::size_t i = 0; i < num_batches; ++i) {
    const auto current_batch_size = std::min(
        batch_size,
        dataset->size() - i*batch_size);
    batched_data.emplace_back(
        data_iter,
        data_iter + static_cast<long>(current_batch_size));
    data_iter += static_cast<long>(current_batch_size);
  }
  return batched_data;
}

std::vector<std::vector<Data *>>
DataHandler::get_batched_training_data(const std::size_t batch_size) const {
  return batch_dataset(training_data_, batch_size);
}

std::vector<std::vector<Data *>>
DataHandler::get_batched_validation_data(const std::size_t batch_size) const {
  return batch_dataset(validation_data_, batch_size);
}

std::vector<std::vector<Data *>>
DataHandler::get_batched_test_data(const std::size_t batch_size) const {
  return batch_dataset(test_data_, batch_size);
}

void DataHandler::normalise_data() const {
  for (const Data *data : *data_array_) {
    std::vector<double> *feature_vector = data->get_feature_vector();
    std::ranges::transform(*feature_vector, feature_vector->begin(),
                           [](auto &val) { return val/255.0; });
  }
}

void DataHandler::print_class_info() const {
  std::cout << "======Class Info======\n";
  for (const auto [k, v] : class_map_) {
    std::cout << "Label: " << static_cast<int>(k) << ", count: " << v << '\n';
  }
}

std::tuple<image_t, label_t>
extract(const Data *d) {
  return std::make_tuple(*d->get_feature_vector(), d->get_label());
}

std::tuple<std::vector<image_t>, std::vector<label_t>>
extract(const data_vec_t *data) {
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
extract(const data_batch_t &batched_data) {
  std::vector<std::vector<image_t>> img_batch;
  std::vector<std::vector<label_t>> lbl_batch;
  img_batch.reserve(batched_data.size());
  lbl_batch.reserve(batched_data.size());

  for (const auto &data_vec : batched_data) {
    auto [imgs, lbls] = extract(&data_vec);
    img_batch.push_back(std::move(imgs));
    lbl_batch.push_back(std::move(lbls));
  }
  return std::make_tuple(std::move(img_batch), std::move(lbl_batch));
}

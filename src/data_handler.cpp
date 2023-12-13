#include <iostream>
#include <fstream>
#include <unordered_set>
#include <random>
#include "../include/data_handler.hpp"

DataHandler::DataHandler() {
  data_array_ = new std::vector<Data*>;
  training_data_ = new std::vector<Data*>;
  validation_data_ = new std::vector<Data*>;
  test_data_ = new std::vector<Data*>;
}

DataHandler::DataHandler(const std::string& image_path, const std::string& label_path)
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

void DataHandler::read_feature_vector(const std::string& path) {
  std::array<uint32_t, 4> header{}; // MAGIC|NUM_IMAGES|ROW_SIZE|COL_SIZE
  std::ifstream bytes_file(path, std::ios::binary);
  read_header(header, bytes_file);

  const size_t image_size = header[2] * header[3];
  image_size_ = image_size;

  while (bytes_file) {
    auto* d = new Data(image_size);
    std::vector<unsigned char> arr(image_size);
    if (!bytes_file.read(reinterpret_cast<char*>(arr.data()), static_cast<long>(image_size))) {
      delete d;
      break;
    }
    d->get_feature_vector()->assign(arr.begin(), arr.end());
    data_array_->emplace_back(d);
  }
  total_length_ = header[1];
  std::cout << "Done extracting " << header[1] << " images\n";
}

template<size_t S>
void DataHandler::read_header(std::array<uint32_t, S>& header, std::ifstream& bytes_file) {
  if (!bytes_file)
    throw std::runtime_error("Failed to open file.\n");

  for (size_t i = 0; i < header.size(); ++i) {
    if (!bytes_file.read(reinterpret_cast<char*>(&header[i]), sizeof(uint32_t)))
      throw std::runtime_error("Failed to read the header at index " + std::to_string(i) + ".\n");
    header[i] = ntohl(header[i]); // convert big->little endian
  }
}

void DataHandler::read_feature_labels(const std::string& path) {
  std::array<uint32_t, 2> header{}; // MAGIC|NUM_LABELS
  std::ifstream bytes_file(path, std::ios::binary);
  read_header(header, bytes_file);

  for (size_t i = 0; i < header[1]; ++i) {
    uint8_t label;
    if (!bytes_file.read(reinterpret_cast<char*>(&label), sizeof(uint8_t)))
      throw std::runtime_error("Failed to read the header at index " + std::to_string(i) + ".\n");
    data_array_->at(i)->set_label(label);
  }
  std::cout << "Done extracting labels\n";
  count_classes();
}

void DataHandler::append_data(const std::vector<size_t>& indices, std::vector<Data*>* dataset, std::vector<Data*>* subset) {
  for (const auto index : indices) {
    subset->emplace_back(dataset->at(index));
  }
}

void DataHandler::split_data() const {

  const size_t train_size = data_array_->size() * TRAIN_SPLIT;
  const size_t validation_size = data_array_->size() * VALIDATION_SPLIT;

  std::vector<size_t> indices(total_length_);
  std::iota(indices.begin(), indices.end(), 0);

  std::random_device rd;
  std::mt19937 g(rd());
  std::ranges::shuffle(indices, g);

  const std::vector<size_t> train_indices(indices.begin(), indices.begin() + static_cast<long>(train_size));
  const std::vector<size_t> validation_indices(indices.begin() + static_cast<long>(train_size),
    indices.begin() + static_cast<long>(train_size + validation_size));
  const std::vector<size_t> test_indices(indices.begin() + static_cast<long>(train_size + validation_size),indices.end());

  append_data(train_indices, data_array_, training_data_);
  append_data(validation_indices, data_array_, validation_data_);
  append_data(test_indices, data_array_, test_data_);

  std::cout << "Training size: " << training_data_->size() << '\n';
  std::cout << "Validation size: " << validation_data_->size() << '\n';
  std::cout << "Test size: " << test_data_->size() << '\n';
}

void DataHandler::count_classes() {
  for (const Data* data: *data_array_) {
    const uint8_t label = data->get_label();
    if (!class_map_.contains(label)) {
      class_map_[label] = 1;
    }
    class_map_[label]++;
  }
}

size_t DataHandler::get_image_size() const { return image_size_; }
size_t DataHandler::num_classes() const { return class_map_.size(); }

const std::vector<Data*>* DataHandler::get_all_data() const {
  return data_array_;
}

const std::vector<Data*>* DataHandler::get_training_data() const { return training_data_; }
const std::vector<Data*>* DataHandler::get_validation_data() const { return validation_data_; }
const std::vector<Data*>* DataHandler::get_test_data() const { return test_data_; }

void DataHandler::normalise_data() const {
  for (const Data* data : *data_array_) {
    std::vector<double>* feature_vector = data->get_feature_vector();
    std::ranges::transform(*feature_vector, feature_vector->begin(),
                           [](auto& val) { return val / 255.0; });
  }
}

void DataHandler::print_class_info() const {
  std::cout << "======Class Info======\n";
  for (const auto [k, v] : class_map_) {
    std::cout << "Label: " << static_cast<int>(k) << ", count: " << v << '\n';
  }
}

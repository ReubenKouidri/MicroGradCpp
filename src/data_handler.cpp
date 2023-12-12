#include <iostream>
#include <fstream>
#include <unordered_set>
#include "../include/data_handler.hpp"

DataHandler::DataHandler() {
  data_array_ = new std::vector<Data*>;
  training_data_ = new std::vector<Data*>;
  validation_data_ = new std::vector<Data*>;
  test_data_ = new std::vector<Data*>;
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
  std::array<uint32_t, 4> header; // MAGIC|NUM_IMAGES|ROW_SIZE|COL_SIZE
  std::ifstream bytes_file(path, std::ios::binary);
  read_header(header, bytes_file);

  const int image_size = header[2] * header[3];
  while (bytes_file) {
    auto* d = new Data(image_size);
    if (!bytes_file.read(reinterpret_cast<char*>(d->get_feature_vector()->data()), image_size)) {
      delete d;
      break;
    }
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
  std::array<uint32_t, 2> header; // MAGIC|NUM_LABELS
  std::ifstream bytes_file(path, std::ios::binary);
  read_header(header, bytes_file);

  for (size_t i = 0; i < header[1]; ++i) {
    uint8_t label;
    if (!bytes_file.read(reinterpret_cast<char*>(&label), sizeof(uint8_t)))
      throw std::runtime_error("Failed to read the header at index " + std::to_string(i) + ".\n");
    data_array_->at(i)->set_label(label);
  }
  std::cout << "Done extracting labels\n";
}

void DataHandler::individual_split(
  std::unordered_set<size_t>& used_indices,
  const size_t split_size,
  std::vector<Data*>* dataset) const {

  size_t count = 0;
  while (count < split_size) {
    if (auto rand_index = rand() % total_length_; !used_indices.contains(rand_index)) {
      dataset->emplace_back(data_array_->at(rand_index)); // emplace back a pointer (Data*)
      used_indices.insert(rand_index);
      count++;
    }
  }
}

void DataHandler::split_data() const {
  std::unordered_set<size_t> used_indices;
  const size_t train_size = data_array_->size() * TRAIN_SPLIT;
  const size_t validation_size = data_array_->size() * VALIDATION_SPLIT;
  const size_t test_size = data_array_->size() * TEST_SPLIT;

  individual_split(used_indices, train_size, training_data_);
  individual_split(used_indices, validation_size, validation_data_);
  individual_split(used_indices, test_size, test_data_);

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
  std::cout << "======Class Info======\n";
  for (const auto [k, v] : class_map_) {
    std::cout << "Label: " << static_cast<int>(k) << ", count: " << v << '\n';
  }
}

size_t DataHandler::image_size() const { return feature_vector_size_; }
size_t DataHandler::num_classes() const { return num_classes_; }

const std::vector<Data*>* const DataHandler::get_training_data() const { return training_data_; }
const std::vector<Data*>* const DataHandler::get_validation_data() const { return validation_data_; }
const std::vector<Data*>* const DataHandler::get_test_data() const { return test_data_; }

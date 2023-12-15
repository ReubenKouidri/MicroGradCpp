#ifndef DATA_HANDLER_HPP
#define DATA_HANDLER_HPP

#include <vector>
#include <map>
#include "data.hpp"

class DataHandler {
  std::vector<Data*> *data_array_;  // all data
  std::vector<Data*> *training_data_;
  std::vector<Data*> *validation_data_;
  std::vector<Data*> *test_data_;

  size_t total_length_ {};
  size_t feature_vector_size_ {};
  size_t image_size_ {};
  std::map<uint8_t, size_t> class_map_ {};

  static constexpr double TRAIN_SPLIT = 0.80;
  static constexpr double VALIDATION_SPLIT = 0.10;
  static constexpr double TEST_SPLIT = 0.10;

  static void append_data(const std::vector<size_t>&, std::vector<Data*>*, std::vector<Data*>*);

  static std::vector<std::vector<Data*>> batch_dataset(const std::vector<Data*>*, size_t);

public:
  DataHandler();
  ~DataHandler();
  DataHandler(const std::string&, const std::string&);

  template<size_t S> static void read_header(std::array<uint32_t, S>&, std::ifstream&);
  void read_feature_vector(const std::string& path);
  void read_feature_labels(const std::string& path);
  void split_data() const;
  void count_classes();
  [[nodiscard]] size_t get_image_size() const;
  [[nodiscard]] size_t num_classes() const;

  [[nodiscard]] const std::vector<Data*>* get_all_data() const;
  [[nodiscard]] const std::vector<Data*>* get_training_data() const;
  [[nodiscard]] const std::vector<Data*>* get_validation_data() const;
  [[nodiscard]] const std::vector<Data*>* get_test_data() const;
  [[nodiscard]] std::vector<std::vector<Data*>> get_batched_training_data(size_t) const;
  [[nodiscard]] std::vector<std::vector<Data*>> get_batched_validation_data(size_t) const;
  [[nodiscard]] std::vector<std::vector<Data*>> get_batched_test_data(size_t) const;

  void normalise_data() const;
  void print_class_info() const;
};

#endif //DATA_HANDLER_HPP
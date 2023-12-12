#ifndef DATA_HANDLER_HPP
#define DATA_HANDLER_HPP

#include <vector>
#include <map>
#include <unordered_set>
#include "data.hpp"

class DataHandler {
  std::vector<Data*> *data_array_;  // all data
  std::vector<Data*> *training_data_;
  std::vector<Data*> *validation_data_;
  std::vector<Data*> *test_data_;

  size_t total_length_ {};
  size_t num_classes_ {};
  size_t feature_vector_size_ {};
  std::map<uint8_t, size_t> class_map_ {};

  static constexpr double TRAIN_SPLIT = 0.80;
  static constexpr double VALIDATION_SPLIT = 0.10;
  static constexpr double TEST_SPLIT = 0.10;

public:
  DataHandler();
  ~DataHandler();

  template<size_t S> static void read_header(std::array<uint32_t, S>&, std::ifstream&);
  void read_feature_vector(const std::string& path);
  void read_feature_labels(const std::string& path);
  void individual_split(std::unordered_set<size_t>&, size_t, std::vector<Data*>*) const;
  void split_data() const;
  void count_classes();
  size_t image_size() const;
  size_t num_classes() const;

  [[nodiscard]] const std::vector<Data*>* const get_training_data() const;
  [[nodiscard]] const std::vector<Data*>* const get_validation_data() const;
  [[nodiscard]] const std::vector<Data*>* const get_test_data() const;
};

#endif //DATA_HANDLER_HPP
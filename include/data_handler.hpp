#ifndef DATA_HANDLER_HPP
#define DATA_HANDLER_HPP

#include <map>
#include <vector>
#include "data.hpp"

static constexpr double TRAIN_SPLIT = 0.80;
static constexpr double VALIDATION_SPLIT = 0.10;

using image_t = std::vector<double>;
using label_t = uint8_t;
using data_vec_t = std::vector<Data *>;
using data_batch_t = std::vector<data_vec_t>;

class DataHandler {
  std::vector<Data *> *data_array_;// all data
  std::vector<Data *> *training_data_;
  std::vector<Data *> *validation_data_;
  std::vector<Data *> *test_data_;

  std::size_t total_length_{};
  std::size_t image_size_{};
  std::map<uint8_t, std::size_t> class_map_{};

  static void append_data(const std::vector<std::size_t> &,
                          std::vector<Data *> *,
                          std::vector<Data *> *);

  static std::vector<std::vector<Data *>> batch_dataset(const std::vector<Data *> *,
                                                        std::size_t);

public:
  DataHandler();
  ~DataHandler();
  [[maybe_unused]] DataHandler(const std::string &, const std::string &);
  DataHandler(const DataHandler &) = delete;
  DataHandler(DataHandler &&) = delete;
  DataHandler &operator=(const DataHandler &) = delete;
  DataHandler &operator=(DataHandler &&) = delete;

  template <std::size_t S>
  static void read_header(std::array<uint32_t, S> &,
                          std::ifstream &);
  void read_feature_vector(const std::string &path);
  void read_feature_labels(const std::string &path);
  void split_data() const;
  void count_classes();
  [[nodiscard]] std::size_t get_image_size() const;
  [[nodiscard]] std::size_t num_classes() const;

  [[nodiscard]] const std::vector<Data *> *get_all_data() const;
  [[nodiscard]] const std::vector<Data *> *get_training_data() const;
  [[nodiscard]] const std::vector<Data *> *get_validation_data() const;
  [[nodiscard]] const std::vector<Data *> *get_test_data() const;
  [[nodiscard]] std::vector<std::vector<Data *>> get_batched_training_data(
    std::size_t) const;
  [[nodiscard]] std::vector<std::vector<Data *>> get_batched_validation_data(
    std::size_t) const;
  [[nodiscard]] std::vector<std::vector<Data *>> get_batched_test_data(std::size_t) const;

  void normalise_data() const;
  void print_class_info() const;
};

std::tuple<image_t, label_t>
extract(const Data *d);

std::tuple<std::vector<image_t>, std::vector<label_t>>
extract(const data_vec_t *data);

std::tuple<std::vector<std::vector<image_t>>, std::vector<std::vector<label_t>>>
extract(const data_batch_t &batched_data);

#endif//DATA_HANDLER_HPP
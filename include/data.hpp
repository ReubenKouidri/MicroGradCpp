#ifndef DATA_HPP
#define DATA_HPP

#include <vector>

class Data {
  std::vector<double> *feature_vector_;
  uint8_t label_{};
 public:
  ~Data() {
    delete feature_vector_;
    feature_vector_ = nullptr;
  }
  Data() { feature_vector_ = new std::vector<double>; }

  explicit Data(const std::size_t sz) {
    feature_vector_ = new std::vector<double>(sz);
  }
  constexpr void set_feature_vector(std::vector<double> * const vec) {
    feature_vector_ = vec;
  }
  constexpr void append_to_feature_vector(const double val) const {
    feature_vector_->emplace_back(val);
  }
  constexpr void set_label(const uint8_t lab) noexcept { label_ = lab; }

  [[nodiscard]] constexpr std::size_t get_feature_vector_size() const {
    return feature_vector_->size();
  }

  [[nodiscard]] constexpr uint8_t get_label() const noexcept { return label_; }

  [[nodiscard]] std::vector<double> *get_feature_vector() const {
    return feature_vector_;
  }
};

#endif //DATA_HPP
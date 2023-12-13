#ifndef DATA_HPP
#define DATA_HPP

#include <vector>

class Data {
  std::vector<double> *feature_vector_;
  uint8_t label_ {};
public:
  Data();
  ~Data();
  explicit Data(size_t);
  void set_feature_vector(std::vector<double>*);
  void append_to_feature_vector(double) const;
  void set_label(uint8_t);

  [[nodiscard]] size_t get_feature_vector_size() const;
  [[nodiscard]] uint8_t get_label() const;
  [[nodiscard]] std::vector<double>* get_feature_vector() const;
};

#endif //DATA_HPP
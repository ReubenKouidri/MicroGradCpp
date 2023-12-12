#include "data.hpp"

Data::Data() { feature_vector_ = new std::vector<uint8_t>; }
Data::Data(const size_t sz) { feature_vector_ = new std::vector<uint8_t>(sz); }

Data::~Data() {
  delete feature_vector_;
  feature_vector_ = nullptr;
}

void Data::set_feature_vector(std::vector<uint8_t>* vect) { feature_vector_ = vect; }
void Data::append_to_feature_vector(const uint8_t val) const { feature_vector_->emplace_back(val); }
void Data::set_label(const uint8_t label) { label_ = label; }
size_t Data::get_feature_vector_size() const { return feature_vector_->size(); }
uint8_t Data::get_label() const { return label_; }
std::vector<uint8_t>* Data::get_feature_vector() const { return feature_vector_; }

#include "../include/data.hpp"

Data::Data() { feature_vector_ = new std::vector<double>; }
Data::Data(const size_t sz) {
  feature_vector_ = new std::vector<double>;
  feature_vector_->reserve(sz);
}

Data::~Data() {
  delete feature_vector_;
  feature_vector_ = nullptr;
}

void Data::set_feature_vector(std::vector<double>* vect) { feature_vector_ = vect; }
void Data::append_to_feature_vector(const double val) const { feature_vector_->emplace_back(val); }
void Data::set_label(const uint8_t label) { label_ = label; }
size_t Data::get_feature_vector_size() const { return feature_vector_->size(); }
uint8_t Data::get_label() const { return label_; }
std::vector<double>* Data::get_feature_vector() const { return feature_vector_; }

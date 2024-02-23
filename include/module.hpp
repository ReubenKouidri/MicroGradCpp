#ifndef MODULE_HPP
#define MODULE_HPP

#include <random>
#include "value.hpp"
#include "utils.hpp"

template <typename T>
using Output = std::vector<Value<T>>;

template <typename T>
using ParamVector = std::vector<std::shared_ptr<Value<T>>>;

template <typename T>
class Module {
 public:
  virtual ~Module() = default;
  Module() = default;
  Module(const Module&) = default;
  Module(Module&&) noexcept = default;
  Module& operator=(const Module&) = default;
  Module& operator=(Module&&) noexcept = default;
  [[nodiscard]] virtual ParamVector<T> get_parameters() const = 0;
};

#endif //MODULE_HPP
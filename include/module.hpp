#ifndef MODULE_HPP
#define MODULE_HPP

#include <random>
#include "value.hpp"
#include "utils.hpp"

template <typename T>
using Output = std::vector<Value<T>>;

template <typename T>
using ParamVector = std::vector<std::shared_ptr<Value<T>>>;

// interface
template <typename T>
class Module {
public:
  virtual ~Module() = default;
  Module() = default;
  Module(const Module& other) = default;
  Module(Module&& other) noexcept = default;
  Module& operator=(const Module& other) = default;
  Module& operator=(Module&& other) noexcept = default;
  [[nodiscard]] virtual ParamVector<T> get_parameters() const = 0;
};

#endif //MODULE_HPP

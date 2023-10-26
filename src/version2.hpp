#ifndef MICROGRAD_VERSION2_HPP
#define MICROGRAD_VERSION2_HPP

#include <vector>
#include <random>

double frand() {
  // static so that only initialised once and reused for all function calls
  static std::mt19937_64 rng(std::random_device{}());
  static std::uniform_real_distribution<double> dist(0.0, 1.0);
  return dist(rng);
}

class Layer {
  typedef std::vector<std::vector<double>> Matrix;
  Matrix weights_;
  Matrix dw_;
  Matrix biases_;
  Matrix db_;
  std::vector<double> xout_ {};
  std::vector<double> dxout_ {};
  std::vector<double> xin_ {};
public:
  Layer(size_t nin, size_t nout) {
    xout_.reserve(nout) ;
    dxout_.reserve(nout);
    weights_.resize(nout);  // nouts == number outputs, i.e. number of rows

    for (auto& w: weights_) {  // init weights
      w.reserve(nin);
      for (int i = 0; i < nin; i++) {
        w.emplace_back(frand());
      }
    }
  }

  const std::vector<double>& forward(const std::vector<double>& xin) {
    xin_ = xin;
    for (int i = 0; i < xout_.size(); i++) {
      for (int j = 0; j < xin_.size(); j++) {
        xout_[i] = weights_[i][j] * xin_[j] + biases_[i][j];
      }
    }
    return xout_;
  }

  void update_weights() {
    for (int i = 0; i < xin_.size(); i++) {
      for (int j = 0; j < xout_.size(); j++) {
        dw_[i][j] = dxout_[i] * xin_[j];
      }
    }
  }

  void update_douts(const std::vector<double>& dx) {
    dxout_ = dx;
  }

  void backwards(const std::vector<double>& grad) {
    update_douts(grad);  // in this order
    update_weights();
  }
};


class MLP {
  std::vector<Layer> layers_;
public:
  MLP(int nin, std::vector<int>& nouts) {
    auto sz = nouts.insert(nouts.begin(), nin);
    for (int i = 0; i < nouts.size(); i++) {
      layers_.emplace_back(sz[i], sz[i + 1]);
    }
  }

  void backwards(const std::vector<double>& initial_grad) {
    // TODO:
    //  - implement backwards
    //  - propagate the grad through the layers.

    layers_[-1].backwards(initial_grad);
  }
};





#endif //MICROGRAD_VERSION2_HPP

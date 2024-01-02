#ifndef INCLUDE_TRAINER_HPP_
#define INCLUDE_TRAINER_HPP_

#include <vector>
#include "components.hpp"
#include "losses.hpp"
#include "optimiser.hpp"

template <typename T, class Loss, class Optimiser>
void train_batched_dataset(const std::shared_ptr<const MLP<T>> &model,
                           const std::vector<typename Loss::batched_input_type> &batched_img_ds,
                           const std::vector<typename Loss::batched_target_type> &batched_tgt_ds,
                           Loss &loss,
                           Optimiser &optimiser,
                           const size_t epochs) {

  const auto num_batches = batched_img_ds.size();
  for (size_t e = 0; e < epochs; e++) {
    double epoch_loss = 0;
    for (size_t i = 0; i < num_batches; i++) {
      train_single_batch(model, batched_img_ds[i], batched_tgt_ds[i], loss, optimiser);
      epoch_loss += loss.get();
      loss.zero();
    }
    std::cout << "Epoch " << e << ": " << "Loss = " << epoch_loss/num_batches
              << '\n';
  }
}

template <typename T, class Loss, class Optimiser>
void train_single_batch(const std::shared_ptr<const MLP<T>> &model,
                        const typename Loss::batched_input_type &img_batch,
                        const typename Loss::batched_target_type &tgt_batch,
                        Loss &loss,
                        Optimiser &optimiser) {
  loss.compute_loss(img_batch, tgt_batch);
  loss.backward();
  optimiser.step();
  model->zero_grad();
}

template <typename T, class Loss, class Optimiser>
void train_single_image(const std::shared_ptr<const MLP<T>> &model,
                        const std::vector<typename Loss::input_type> &imgs,
                        const std::vector<typename Loss::target_type> &tgts,
                        Loss &loss,
                        Optimiser &optimiser,
                        const size_t epochs) {
  for (size_t e = 0; e < epochs; e++) {
    for (size_t i{0}; i < imgs.size(); i++) {
      loss.compute_loss(imgs[i], tgts[i]);
      loss.backward();
      optimiser.step();
      model->zero_grad();
      std::cout << "Loss = " << loss.get() << '\n';
      loss.zero();
    }
  }
}

template <typename T, class Loss, class Input_Tp, class Target_Tp, class Optimiser>
void train_model(const std::shared_ptr<const MLP<T>> &model,
                 const Input_Tp &inputs,
                 const Target_Tp &targets,
                 Loss &loss,
                 Optimiser &optimiser,
                 const size_t epochs) {
  const auto num_samples = inputs.size();
  for (size_t e = 0; e < epochs; e++) {
    double epoch_loss = 0;
    for (size_t i = 0; i < num_samples; i++) {
      loss.compute_loss(inputs, targets);
      epoch_loss += loss.get();
      loss.backward();
      optimiser.step();
      model->zero_grad();
      loss.zero();
    }
    std::cout << "Epoch " << e << ": "
              << "Loss = " << epoch_loss/num_samples
              << '\n';
  }
}


#endif //INCLUDE_TRAINER_HPP_
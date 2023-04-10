
/*
 * torch-gobject/nn/options/torch-nn-loss-reduction-mode-internal.h
 *
 * GridSample mode internal functionality
 *
 * Copyright (C) 2021 Sam Spilsbury.
 *
 * torch-gobject is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or
 * (at your option) any later version.
 *
 * torch-gobject is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along
 * with torch-gobject; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#pragma once

#include <torch/enum.h>
#include <torch-gobject/torch-util.h>
#include <torch-gobject/nn/options/torch-nn-loss-reduction-mode.h>

namespace
{
  /* This is probably not the best place for this helper to live,
   * but it fits for now. */
  template <class... Args>
  struct VariantCaster
  {
    VariantCaster(c10::variant<Args...> &&v) :
      v(v)
    {
    }

    /* We can also accept another variant if it can be casted into our variant type */
    template <class... FromArgs>
    VariantCaster(c10::variant<FromArgs...> &&v) :
      v(variant_cast(std::forward(v)))
    {
    }

    c10::variant<Args...> v;

    template <class... ToArgs>
    operator c10::variant<ToArgs...>() const
    {
      return c10::visit([](auto&& arg) -> c10::variant<ToArgs...> { return arg ; }, v);
    }
  };

  template <class... Args>
  auto variant_cast(c10::variant<Args...> const & v) -> VariantCaster<Args...>
  {
    return {v};
  }

  /* The LossVariantType is the true return type, the LossVariantCaster can "decay" into any
   * other variant type which is a superset of this type. */
  typedef c10::variant<torch::enumtype::kNone, torch::enumtype::kMean, torch::enumtype::kSum> LossVariantType;
  typedef c10::variant<torch::enumtype::kNone, torch::enumtype::kBatchMean, torch::enumtype::kMean, torch::enumtype::kSum> ExpandedLossVariantType;

  typedef VariantCaster<torch::enumtype::kNone, torch::enumtype::kBatchMean, torch::enumtype::kMean, torch::enumtype::kSum> InLossVariantCaster;
  typedef VariantCaster<torch::enumtype::kNone, torch::enumtype::kMean, torch::enumtype::kSum> OutLossVariantCaster;
}

OutLossVariantCaster
torch_nn_loss_reduction_mode_to_real_loss_reduction_mode (TorchNNLossReductionMode mode)
{
    switch (mode)
    {
      case TORCH_NN_LOSS_REDUCTION_MODE_MEAN:
        return OutLossVariantCaster(LossVariantType(torch::enumtype::kMean()));
      case TORCH_NN_LOSS_REDUCTION_MODE_SUM:
        return OutLossVariantCaster(LossVariantType(torch::enumtype::kSum()));
      case TORCH_NN_LOSS_REDUCTION_MODE_NONE:
        return OutLossVariantCaster(LossVariantType(torch::enumtype::kNone()));
      default:
        throw std::logic_error("Invalid loss reduction mode");
    }
}

TorchNNLossReductionMode
torch_nn_loss_reduction_mode_from_real_loss_reduction_mode (InLossVariantCaster &&caster)
{
  ExpandedLossVariantType mode(caster);

  if (c10::get_if<torch::enumtype::kMean> (&mode)) {
    return TORCH_NN_LOSS_REDUCTION_MODE_MEAN;
  }

  if (c10::get_if<torch::enumtype::kSum> (&mode)) {
    return TORCH_NN_LOSS_REDUCTION_MODE_SUM;
  }

  if (c10::get_if<torch::enumtype::kNone> (&mode)) {
    return TORCH_NN_LOSS_REDUCTION_MODE_NONE;
  }

  throw std::logic_error("Invalid loss reduction mode");
}

namespace torch
{
  namespace gobject
  {
    template <>
    struct ConversionTrait<TorchNNLossReductionMode>
    {
      typedef OutLossVariantCaster real_type;

      /* With variants we might want to cast the output of this function
       *
       * we want this to be compatible with both:
       * - c10::variant<torch::enumtype::kNone, torch::enumtype::kMean, torch::enumtype::kSum>
       * - c10::variant<torch::enumtype::kNone, torch::enumtype::kBatchMean, torch::enumtype::kMean, torch::enumtype::kSum>
       */
      static constexpr auto from = torch_nn_loss_reduction_mode_to_real_loss_reduction_mode;
      static constexpr auto to = torch_nn_loss_reduction_mode_from_real_loss_reduction_mode;
    };

    template<>
    struct ReverseConversionTrait<ExpandedLossVariantType>
    {
      typedef TorchNNLossReductionMode gobject_type;
    };

    template<>
    struct ReverseConversionTrait<LossVariantType>
    {
      typedef TorchNNLossReductionMode gobject_type;
    };
  }
}

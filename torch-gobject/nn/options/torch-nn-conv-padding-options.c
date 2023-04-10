/*
 * torch-gobject/nn/options/torch-nn-conv-padding-options.c
 *
 * Convolution operator padding options
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

#include <torch-gobject/nn/options/torch-nn-conv-padding-type.h>
#include <torch-gobject/nn/options/torch-nn-conv-padding-options.h>

struct _TorchNNConvPaddingOptions {
  TorchNNConvPaddingType  padding_type;
  GArray                 *padding_config;
};

struct _TorchNNConvPaddingOptions1D {
  TorchNNConvPaddingOptions base;
};

struct _TorchNNConvPaddingOptions2D {
  TorchNNConvPaddingOptions base;
};

struct _TorchNNConvPaddingOptions3D {
  TorchNNConvPaddingOptions base;
};

/**
 * torch_nn_conv_padding_options_new:
 * @padding_type: A #TorchNNConvPaddingType , either %TORCH_NN_CONV_PADDING_TYPE_SPECIFIED
 *                where @padding_config is set or another value.
 * @padding_config: (array length=padding_config_length): The padding configuration array
 * @padding_config_length: Length of the padding config array.
 *
 * Returns: (transfer full): A new #TorchNNConvPaddingOptions
 */
TorchNNConvPaddingOptions *
torch_nn_conv_padding_options_new (TorchNNConvPaddingType  padding_type,
                                   int64_t                *padding_config,
                                   size_t                  padding_config_length)
{
  g_assert (
    (padding_type == TORCH_NN_CONV_PADDING_TYPE_SPECIFIED && padding_config != NULL) ||
    (padding_type != TORCH_NN_CONV_PADDING_TYPE_SPECIFIED && padding_config == NULL)
  );

  TorchNNConvPaddingOptions *opts = g_new0 (TorchNNConvPaddingOptions, 1);
  opts->padding_type = padding_type;
  opts->padding_config = padding_config != NULL ? g_array_sized_new (FALSE, FALSE, sizeof (int64_t), padding_config_length) : NULL;

  opts->padding_config = g_array_insert_vals (opts->padding_config, 0, padding_config, padding_config_length);

  return g_steal_pointer (&opts);
}

/**
 * torch_nn_conv_padding_options_1d_new:
 * @padding_type: A #TorchNNConvPaddingType , either %TORCH_NN_CONV_PADDING_TYPE_SPECIFIED
 *                where @padding_config is set or another value.
 * @padding_config: (array fixed-size=1): The padding configuration array
 *
 * Returns: (transfer full): A new #TorchNNConvPaddingOptions1D
 */
TorchNNConvPaddingOptions1D *
torch_nn_conv_padding_options_1d_new (TorchNNConvPaddingType  padding_type,
                                      int64_t                *padding_config)
{
  return (TorchNNConvPaddingOptions1D *) torch_nn_conv_padding_options_new (padding_type, padding_config, 1);
}

/**
 * torch_nn_conv_padding_options_2d_new:
 * @padding_type: A #TorchNNConvPaddingType , either %TORCH_NN_CONV_PADDING_TYPE_SPECIFIED
 *                where @padding_config is set or another value.
 * @padding_config: (array fixed-size=2): The padding configuration array
 *
 * Returns: (transfer full): A new #TorchNNConvPaddingOptions2D
 */
TorchNNConvPaddingOptions2D *
torch_nn_conv_padding_options_2d_new (TorchNNConvPaddingType  padding_type,
                                      int64_t                *padding_config)
{
  return (TorchNNConvPaddingOptions2D *) torch_nn_conv_padding_options_new (padding_type, padding_config, 2);
}

/**
 * torch_nn_conv_padding_options_3d_new:
 * @padding_type: A #TorchNNConvPaddingType , either %TORCH_NN_CONV_PADDING_TYPE_SPECIFIED
 *                where @padding_config is set or another value.
 * @padding_config: (array fixed-size=1): The padding configuration array
 *
 * Returns: (transfer full): A new #TorchNNConvPaddingOptions3D
 */
TorchNNConvPaddingOptions3D *
torch_nn_conv_padding_options_3d_new (TorchNNConvPaddingType  padding_type,
                                      int64_t                *padding_config)
{
  return (TorchNNConvPaddingOptions3D *) torch_nn_conv_padding_options_new (padding_type, padding_config, 3);
}

/**
 * torch_nn_conv_padding_options_copy:
 * @opts: (transfer none): A #TorchNNConvPaddingOptions to copy.
 *
 * Returns: (transfer full): A new #TorchNNConvPaddingOptions which is a copy of @opts
 */
TorchNNConvPaddingOptions *
torch_nn_conv_padding_options_copy (TorchNNConvPaddingOptions *opts)
{
  return torch_nn_conv_padding_options_new (opts->padding_type, (int64_t *) opts->padding_config->data, opts->padding_config->len);
}

static void free_array (GArray *array)
{
  g_array_free (array, TRUE);
}

/**
 * torch_nn_conv_padding_options_free:
 * @opts: A #TorchNNConvPaddingOptions to free.
 */
void
torch_nn_conv_padding_options_free (TorchNNConvPaddingOptions *opts)
{
  g_clear_pointer (&opts->padding_config, free_array);
  g_clear_pointer (&opts, g_free);
}

G_DEFINE_BOXED_TYPE (TorchNNConvPaddingOptions,
                     torch_nn_conv_padding_options,
                     (GBoxedCopyFunc) torch_nn_conv_padding_options_copy,
                     (GBoxedFreeFunc) torch_nn_conv_padding_options_free)

G_DEFINE_BOXED_TYPE (TorchNNConvPaddingOptions1D,
                     torch_nn_conv_padding_options_1d,
                     (GBoxedCopyFunc) torch_nn_conv_padding_options_copy,
                     (GBoxedFreeFunc) torch_nn_conv_padding_options_free)

G_DEFINE_BOXED_TYPE (TorchNNConvPaddingOptions2D,
                     torch_nn_conv_padding_options_2d,
                     (GBoxedCopyFunc) torch_nn_conv_padding_options_copy,
                     (GBoxedFreeFunc) torch_nn_conv_padding_options_free)

G_DEFINE_BOXED_TYPE (TorchNNConvPaddingOptions3D,
                     torch_nn_conv_padding_options_3d,
                     (GBoxedCopyFunc) torch_nn_conv_padding_options_copy,
                     (GBoxedFreeFunc) torch_nn_conv_padding_options_free)

/**
 * torch_nn_conv_padding_options_get_padding_type:
 * @opts: (transfer none): A #TorchNNConvPaddingOptions
 *
 * Returns: The #TorchNNConvPaddingType for this @opts
 */
TorchNNConvPaddingType
torch_nn_conv_padding_options_get_padding_type (TorchNNConvPaddingOptions *opts)
{
  return opts->padding_type;
}

/**
 * torch_nn_conv_padding_options_get_padding_config:
 * @opts: (transfer none): A #TorchNNConvPaddingOptions
 *
 * Returns: (transfer none) (element-type gint64): The #GArray of padding configuration for this
 *                                                 @opts or %NULL if the padding_type member was set
 *                                                 to other than %TORCH_NN_CONV_PADDING_TYPE_SPECIFIED
 */
GArray *
torch_nn_conv_padding_options_get_padding_config (TorchNNConvPaddingOptions *opts)
{
  return opts->padding_config;
}
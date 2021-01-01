/*
 * torch-gobject/torch-tensor.cpp
 *
 * Tensor abstraction for data to be passed to a tensor.
 *
 * Copyright (C) 2020 Sam Spilsbury.
 *
 * libanimation is free software: you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 2.1 of the
 * License, or (at your option) any later version.
 *
 * libanimation is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with eos-companion-app-service.  If not, see
 * <http://www.gnu.org/licenses/>.
 */

#include <stdexcept>
#include <vector>

#include <ATen/Tensor.h>

#include <gio/gio.h>

#include <torch-gobject/torch-device.h>
#include <torch-gobject/torch-device-internal.h>
#include <torch-gobject/torch-errors.h>
#include <torch-gobject/torch-tensor.h>
#include <torch-gobject/torch-tensor-internal.h>
#include <torch-gobject/torch-util.h>

struct _TorchTensor
{
  GObject parent_instance;
};

typedef struct _TorchTensorPrivate
{
  torch::Tensor *internal;

  GVariant    *construction_data;
  GList       *construction_dims;
  gboolean     is_constructed : 1;
} TorchTensorPrivate;

static void initable_iface_init (GInitableIface *iface);

G_DEFINE_TYPE_WITH_CODE (TorchTensor, torch_tensor, G_TYPE_OBJECT,
                         G_ADD_PRIVATE (TorchTensor)
                         G_IMPLEMENT_INTERFACE (G_TYPE_INITABLE, initable_iface_init))
#define TORCH_TENSOR_GET_PRIVATE(a) static_cast <TorchTensorPrivate *> (torch_tensor_get_instance_private((a)))

enum {
  PROP_0,
  PROP_DATA,
  PROP_DIMS,
  PROP_DTYPE,
  NPROPS
};

static GParamSpec *torch_tensor_props [NPROPS] = { NULL, };

namespace
{
  template <typename T>
  void safe_delete (T *t)
  {
    delete t;
  }

  /* XXX: Its not entirely clear to me why,
   *      but if we return an IntArrayRef here, we crash
   *      because at::List doesn't make a copy of the underlying
   *      memory, and the constructor does not take an rvalue
   *      reference, so the move never happens. */
  template <typename Source>
  std::vector<int64_t> int_list_from_g_array (GArray *array)
  {
    std::vector <int64_t> vec;
    size_t n_elements = array != NULL ? array->len : 0;
    const Source *array_data = array != NULL ? reinterpret_cast <Source *> (array->data) : NULL;
    vec.reserve (n_elements);

    for (size_t i = 0; i < n_elements; ++i)
      vec.push_back (array_data[i]);

    return vec;
  }

  template <typename Target>
  GArray * g_array_from_int_list (torch::IntArrayRef const &list)
  {
    g_autoptr (GArray) array = g_array_sized_new (FALSE, FALSE, sizeof (Target), list.size ());
    Target            *array_data = reinterpret_cast <Target *> (array->data);
    const int64_t     *array_ref_data = list.data ();

    // Because some language bindings rely on types being
    // big enough to fit in GIArgument and we can't break ABI,
    // downcasting may be necessary in some cases
    for (size_t i = 0; i < array->len; ++i)
      array_data[i] = (Target) (array_ref_data[i]);

    return static_cast <GArray *> (g_steal_pointer (&array));
  }

  template <typename Source>
  std::vector<int64_t> int_list_from_g_list (GList *list)
  {
    std::vector <int64_t> vec;

    for (GList *link = list; link != NULL; link = link->next)
      vec.push_back (GPOINTER_TO_INT (link->data));

    return vec;
  }

  template <typename Target>
  GList * g_list_from_int_list (torch::IntArrayRef const &array_ref)
  {
    g_autoptr (GList) list = NULL;
    const int64_t     *array_ref_data = array_ref.data ();

    // Because some language bindings rely on types being
    // big enough to fit in GIArgument and we can't break ABI,
    // downcasting may be necessary in some cases
    for (size_t i = 0; i < array_ref.size(); ++i)
      list = g_list_prepend (list, GINT_TO_POINTER (array_ref_data[i]));

    list = g_list_reverse (list);
    return static_cast <GList *> (g_steal_pointer (&list));
  }

  class InvalidVariantTypeError : public std::logic_error
  {
    public:
      InvalidVariantTypeError (GVariantType const *variant_type) :
        std::logic_error::logic_error (InvalidVariantTypeError::format_error (variant_type))
      {
      }

    private:
      static inline std::string format_error (GVariantType const *variant_type)
      {
        std::stringstream ss;
        ss << "Cannot convert GVariantType "
           << g_variant_type_peek_string (variant_type)
           << " to ScalarType";
        return ss.str ();
      }
  };

  class InvalidScalarTypeError : public std::logic_error
  {
    public:
      InvalidScalarTypeError (c10::ScalarType const &scalar_type) :
        std::logic_error::logic_error (InvalidScalarTypeError::format_error (scalar_type))
      {
      }

    private:
      static inline std::string format_error (c10::ScalarType const &scalar_type)
      {
        std::stringstream ss;
        ss << "Cannot handle scalar type " << scalar_type;
        return ss.str ();
      }
  };

  GVariantType const * scalar_type_to_g_variant_type (c10::ScalarType scalar_type)
  {
    /* XXX: We do not support float tensors
     *      at the moment as GVariant doesn't
     *      support floats. */
    if (scalar_type == torch::kFloat64) {
      return G_VARIANT_TYPE_DOUBLE;
    } else if (scalar_type == torch::kFloat) {
      return G_VARIANT_TYPE_DOUBLE;
    }else if (scalar_type == torch::kInt64) {
      return G_VARIANT_TYPE_INT64;
    } else {
      throw InvalidScalarTypeError (scalar_type);
    }
  }

  size_t scalar_type_to_element_size (c10::ScalarType scalar_type)
  {
    if (scalar_type == torch::kFloat64) {
      return sizeof (double);
    } else if (scalar_type == torch::kInt64) {
      return sizeof (int64_t);
    } else {
      throw InvalidScalarTypeError (scalar_type);
    }
  }

  template <typename T>
  std::vector <T>
  append_to_vector (std::vector <T> &&vec, T &&v)
  {
    std::vector vec_out (vec);
    vec_out.push_back(v);
    return vec_out;
  }

  std::tuple <GVariantType const *, std::vector <int64_t>> ascertain_underlying_type_and_dimensions (GVariant *array_variant)
  {
    GVariantType const *array_variant_type = G_VARIANT_TYPE (g_variant_get_type_string (array_variant));
    if (!g_variant_type_equal (array_variant_type, G_VARIANT_TYPE ("av")))
      {
        g_assert (g_variant_type_is_container (array_variant_type));
        return std::make_tuple (array_variant_type,
                                append_to_vector (std::vector <int64_t> (),
                                                  static_cast <int64_t> (g_variant_n_children (array_variant))));
      }

    g_autoptr(GVariant) child_variant = g_variant_ref_sink (g_variant_get_child_value (array_variant, 0));
    g_autoptr(GVariant) child_array = g_variant_ref_sink (g_variant_get_variant (child_variant));

    GVariantType const *variant_type;
    std::vector <int64_t> dimension_vec;

    std::tie (variant_type, dimension_vec) = ascertain_underlying_type_and_dimensions (child_array);

    return std::make_tuple (variant_type,
                            append_to_vector (std::move (dimension_vec),
                                              static_cast <int64_t> (g_variant_n_children (array_variant))));
  }

  template <typename T>
  void iterate_and_assign_to_tensor (torch::Tensor &tensor,
                                     const char    *type_string,
                                     GVariant      *array_variant)
  {
    T scalar;
    GVariantIter iter;
    size_t       count = 0;

    g_variant_iter_init (&iter, array_variant);

    while (g_variant_iter_next (&iter, type_string, &scalar))
      {
        tensor[count++] = scalar;
      }
  }

  void set_tensor_data_from_nested_variant_arrays (torch::Tensor       &tensor,
                                                   GVariant            *array_variant,
                                                   GVariantType  const *underlying_type)
  {
    /* Base case */
    if (!g_variant_is_of_type (array_variant, G_VARIANT_TYPE ("av")))
      {
        g_assert (g_variant_type_is_container (G_VARIANT_TYPE (g_variant_get_type_string (array_variant))));
        g_assert (g_variant_is_of_type (array_variant, underlying_type));

        const char   *type_string = (g_variant_type_peek_string (underlying_type) + 1);

        if (g_variant_type_equal (underlying_type, G_VARIANT_TYPE ("ad")))
          {
            iterate_and_assign_to_tensor <double> (tensor, type_string, array_variant);
          }
        else if (g_variant_type_equal (underlying_type, G_VARIANT_TYPE ("ax")))
          {
            iterate_and_assign_to_tensor <int64_t> (tensor, type_string, array_variant);
          }
        else
          {
            throw InvalidVariantTypeError (underlying_type);
          }

        return;
      }

    /* Recursive case */
    GVariantIter iter;
    GVariant     *unowned_child_array;
    size_t       tensor_index = 0;

    g_variant_iter_init (&iter, array_variant);
    while (g_variant_iter_next (&iter, "v", &unowned_child_array))
      {
        g_autoptr(GVariant) child_array = g_variant_ref_sink (unowned_child_array);

        torch::Tensor child_tensor (tensor[tensor_index]);
        set_tensor_data_from_nested_variant_arrays (child_tensor, child_array, underlying_type);

        ++tensor_index;
      }
  }

  torch::Tensor new_tensor_from_nested_gvariants (GVariant *array_variant)
  {
    GVariantType const *underlying_type;
    std::vector <int64_t> dimensions;

    std::tie (underlying_type, dimensions) = ascertain_underlying_type_and_dimensions (array_variant);
    std::reverse (dimensions.begin (), dimensions.end ());

    torch::Tensor tensor = torch::zeros (torch::IntArrayRef (dimensions)).cpu ();
    set_tensor_data_from_nested_variant_arrays (tensor, array_variant, underlying_type);

    return tensor;
  }

  GVariant * serialize_tensor_data_to_nested_gvariants (torch::Tensor const &tensor)
  {
    /* Base case, only a single dimension left */
    if (tensor.dim () == 1)
      {
        size_t sz = tensor.sizes ()[0];
        if (tensor.scalar_type () == torch::kFloat)
          {
            /* Different logic, we need to convert the array of floats
             * into an array of doubles first */
            std::vector <double> data (sz);
            const float *tensor_data = static_cast <float *> (tensor.data_ptr ()); 

            for (size_t i = 0; i < sz; ++i)
              {
                data[i] = tensor_data[i];
              }

            return g_variant_new_fixed_array (scalar_type_to_g_variant_type (tensor.scalar_type ()),
                                              tensor_data,
                                              sz,
                                              sizeof (double));
          }

        return g_variant_new_fixed_array (scalar_type_to_g_variant_type (tensor.scalar_type ()),
                                          tensor.data_ptr (),
                                          sz,
                                          scalar_type_to_element_size (tensor.scalar_type ()));
      }

    /* Recursive case: Build a new array-of-variants
     * by looping through the current dimension and
     * getting arrays from that. */
    g_auto(GVariantBuilder) builder = G_VARIANT_BUILDER_INIT (G_VARIANT_TYPE ("av"));
    g_variant_builder_init (&builder, G_VARIANT_TYPE ("av"));
    size_t size = tensor.sizes ()[0];

    for (size_t i = 0; i < size; ++i)
      {
        g_variant_builder_add (&builder,
                               "v",
                               serialize_tensor_data_to_nested_gvariants (tensor[i]));
      }

    return g_variant_builder_end (&builder);
  }
}

torch::Tensor &
torch_tensor_get_real_tensor (TorchTensor *tensor)
{
  TorchTensorPrivate *priv = TORCH_TENSOR_GET_PRIVATE (tensor);

  return *priv->internal;
}

gboolean
torch_tensor_init_internal (TorchTensor  *tensor,
                            GError      **error)
{
  g_return_val_if_fail (error == NULL || *error == NULL, FALSE);

  TorchTensorPrivate *priv = TORCH_TENSOR_GET_PRIVATE (tensor);

  /* Even though we have a check in torch_tensor_initable_init,
   * check again here to avoid the vfunc calls */
  if (!priv->internal)
    return g_initable_init (G_INITABLE (tensor), NULL, error);
}

/**
 * torch_tensor_get_dims:
 * @tensor: A #TorchTensor
 * @error: A #GError
 *
 * Get the dimensionality of the tensor in the form of an array
 * of integer values.
 *
 * Arrays can be N-dimensional, as indicated by the number of
 * elements in the array. For instance, a Tensor with dimension
 * [3, 4, 5] has 3 rows, 4 columns and 5 stacks.
 *
 * Returns: (element-type guint) (transfer full): A #GList of integer
 *          values representing the dimensionality of the array,
 *          or %NULL with @error set on failure.
 */
GList *
torch_tensor_get_dims (TorchTensor  *tensor,
                       GError      **error)
{
  TorchTensorPrivate *priv =
    static_cast <TorchTensorPrivate *> (torch_tensor_get_instance_private (tensor));

  try
    {
      return g_list_from_int_list <unsigned int> (priv->internal->sizes ());
    }
  catch (std::exception const &e)
    {
      return reinterpret_cast <GList *> (set_error_from_exception (e,
                                                                   G_IO_ERROR,
                                                                   G_IO_ERROR_FAILED,
                                                                   error));
    }

  return g_list_copy (priv->construction_dims);
}

/**
 * torch_tensor_set_dims:
 * @tensor: A #TorchTensor
 * @dims: (element-type guint): A #GList of integer values
 *                              representing the dimensionality of the array.
 *
 * Set the dimensionality of the tensor in the form of an array
 * of integer values. If the change in dimensionality results
 * in fewer total array cells than before, then the Tensor
 * data will be truncated. If the change in dimensionality results
 * in more total array cells than before, then the Tensor
 * data will be padded at the end with uninitialized data.
 *
 * Arrays can be N-dimensional, as indicated by the number of
 * elements in the array. For instance, a Tensor with dimension
 * [3, 4, 5] has 3 rows, 4 columns and 5 stacks.
 *
 * Note that this will cause the tensor storage to be re-allocated
 * in-place and will throw away gradients, so is most likely
 * not what you want in normal operation. Instead consider
 * using %torch_tensor_reshape.
 *
 * Returns: %TRUE on success, %FALSE with @error set on failure.
 */
gboolean
torch_tensor_set_dims (TorchTensor  *tensor,
                       GList        *dims,
                       GError      **error)
{
  TorchTensorPrivate *priv =
    static_cast <TorchTensorPrivate *> (torch_tensor_get_instance_private (tensor));

  if (!priv->is_constructed)
    {
      g_clear_pointer (&priv->construction_dims, g_list_free);
      priv->construction_dims = dims != NULL ? g_list_copy (dims) : NULL;
      return TRUE;
    }

  if (!torch_tensor_init_internal (tensor, error))
    return FALSE;

  try
    {
      priv->internal->resize_ (torch::IntArrayRef (int_list_from_g_list <unsigned int> (dims)));
    }
  catch (std::exception const &e)
    {
      return (gboolean) (set_error_from_exception (e,
                                                   G_IO_ERROR,
                                                   G_IO_ERROR_FAILED,
                                                   error));
    }
}

/**
 * torch_tensor_get_tensor_data:
 * @tensor: A tensor to get the data for.
 * @error: A #GError
 *
 * Return the underlying data for a tensor as an array of variants
 * (av), where each variant in the array is itself an array
 * array of variants or an array of a particular datatype
 * (d|x|f).
 *
 * The level of nesting of array-variants corresponds to
 * the number of dimensions in the tensor. For instance, a 2D
 * tensor will have an array of arrays of (d|x|f). It is the programmer's
 * responsibility to ensure that the returned variant is decoded
 * properly, both in terms of its nesting and its underlying
 * datatype.
 *
 * Note that calling this function causes a copy of the data into
 * the variant, as opposed to just making a view. This can be as
 * expensive as copying from GPU to CPU, so it should be used
 * sparingly.
 *
 * Returns: (transfer full): A floating reference to a new
 *          #GVariant containing the tensor data
 *          or %NULL with @error set on failure.
 */
GVariant *
torch_tensor_get_tensor_data (TorchTensor  *tensor,
                              GError      **error)
{
  TorchTensorPrivate *priv =
    static_cast <TorchTensorPrivate *> (torch_tensor_get_instance_private (tensor));

  if (!torch_tensor_init_internal (tensor, error))
    return NULL;

  try
    {
      return g_variant_ref_sink (serialize_tensor_data_to_nested_gvariants (*priv->internal));
    }
  catch (InvalidScalarTypeError const &e)
    {
      return reinterpret_cast <GVariant *> (set_error_from_exception (e,
                                                                      TORCH_ERROR,
                                                                      TORCH_ERROR_INVALID_DATA_TYPE,
                                                                      error));
    }
}

/**
 * torch_tensor_set_data:
 * @tensor: A tensor to set the data on
 * @data: (transfer none): A #GVariant of type "av" containing
 *        an array of variants according to the schema
 *        specified in %torch_tensor_get_data.
 *
 * The tensor will be automatically resized and adopt
 * the dimensionality of the nested array of variants. It
 * is the programmer's responsibility to ensure that
 * sub-array sizes are consistent between sub-arrays
 * of the same dimension and that the underlying datatype
 * is consistent between all sub-arrays.
 *
 * PyTorch will likely copy the contents of the array
 * either into CPU memory or GPU memory as a result of
 * calling this function, so it should be used seldomly.
 *
 * Returns: %TRUE on success or %FALSE with @error set on failure.
 */
gboolean
torch_tensor_set_data (TorchTensor  *tensor,
                       GVariant     *data,
                       GError      **error)
{
  TorchTensorPrivate *priv =
    static_cast <TorchTensorPrivate *> (torch_tensor_get_instance_private (tensor));

  /* %NULL data is ignored */
  if (data == nullptr)
    return TRUE;

  if (!priv->is_constructed)
    {
      g_clear_pointer (&priv->construction_data, g_variant_unref);
      priv->construction_data = g_variant_ref (data);
    }

  if (!torch_tensor_init_internal (tensor, error))
    return FALSE;

  try
    {
      priv->internal->set_data (new_tensor_from_nested_gvariants (data));
    }
  catch (InvalidVariantTypeError &e)
    {
      return (gboolean) (set_error_from_exception (e,
                                                   TORCH_ERROR,
                                                   TORCH_ERROR_INVALID_DATA_TYPE,
                                                   error));
    }
  catch (std::exception const &e)
    {
      return (gboolean) (set_error_from_exception (e,
                                                   G_IO_ERROR,
                                                   G_IO_ERROR_FAILED,
                                                   error));
    }

  return TRUE;
}

/**
 * torch_tensor_get_dtype:
 * @tensor: (transfer none): A #TorchTensor
 * @error: A #GError
 *
 * Get the data type of a TorchTensor
 *
 * Returns: A #GType with the internal data type of this tensor,
 *          0 on failure with @error set.
 */
GType
torch_tensor_get_dtype (TorchTensor  *tensor,
                        GError      **error)
{
  TorchTensorPrivate *priv = TORCH_TENSOR_GET_PRIVATE (tensor);

  if (!torch_tensor_init_internal (tensor, error))
    return static_cast <GType> (0);

  return torch_gtype_from_scalar_type (priv->internal->scalar_type ());
}

static gboolean
torch_tensor_initable_init (GInitable     *initable,
                            GCancellable  *cancellable,
                            GError       **error)
{
  TorchTensor        *tensor = reinterpret_cast <TorchTensor *> (initable);
  TorchTensorPrivate *priv = TORCH_TENSOR_GET_PRIVATE (tensor);

  /* Already initialized, skip */
  if (priv->internal)
    return TRUE;

  try
    {
      if (priv->construction_data)
        {
          priv->internal = new torch::Tensor (new_tensor_from_nested_gvariants (priv->construction_data));

          if (priv->construction_dims)
            {
              torch_tensor_set_dims (tensor, priv->construction_dims, error);
            }

        }
      else
        {
          priv->internal = new torch::Tensor (); 
        }

      /* Once we've constructed the internal, everything gets moved to
       * the internal tensor (one canonical copy), so we can clear the construct
       * properties that we had in the meantime */
      g_clear_pointer (&priv->construction_dims, g_list_free);
      g_clear_pointer (&priv->construction_data, g_variant_unref);
      priv->is_constructed = TRUE;
    }
  catch (const std::exception &exp)
    {
      g_set_error (error,
                   G_IO_ERROR,
                   G_IO_ERROR_FAILED,
                   exp.what(),
                   nullptr);
      return FALSE;
    }

  return TRUE;
}

static void
initable_iface_init (GInitableIface *iface)
{
  iface->init = torch_tensor_initable_init;
}

static void
torch_tensor_init (TorchTensor *tensor)
{
}

static void
torch_tensor_finalize (GObject *object)
{
  TorchTensor *tensor = TORCH_TENSOR (object);
  TorchTensorPrivate *priv = TORCH_TENSOR_GET_PRIVATE (tensor);

  if (priv->internal)
    {
      delete priv->internal;
      priv->internal = nullptr;
    }

  g_clear_pointer (&priv->construction_dims, g_list_free);
  g_clear_pointer (&priv->construction_data, g_variant_unref);
}

static void
torch_tensor_constructed (GObject *object)
{
  TorchTensor        *tensor = TORCH_TENSOR (object);
  TorchTensorPrivate *priv = TORCH_TENSOR_GET_PRIVATE (tensor);

  priv->is_constructed = TRUE;
}

static void
torch_tensor_get_property (GObject      *object,
                           unsigned int  prop_id,
                           GValue       *value,
                           GParamSpec   *pspec)
{
  TorchTensor *tensor = TORCH_TENSOR (object);

  /* Properties only get set on construction */
  switch (prop_id)
    {
      case PROP_DATA:
        g_value_set_variant (value,
                             call_and_warn_about_gerror ("get property 'data'",
                                                         torch_tensor_get_tensor_data,
                                                         tensor));
        break;
      case PROP_DIMS:
        g_value_set_boxed (value,
                           call_and_warn_about_gerror ("get property 'dimensions'",
                                                       torch_tensor_get_dims,
                                                       tensor));
        break;
      case PROP_DTYPE:
        g_value_set_gtype (value,
                           call_and_warn_about_gerror ("get property 'dtype'",
                                                       torch_tensor_get_dtype,
                                                       tensor));
        break;
      default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
        break;
    }
}

static void
torch_tensor_set_property (GObject      *object,
                            unsigned int  prop_id,
                            const GValue *value,
                            GParamSpec   *pspec)
{
  TorchTensor *tensor = TORCH_TENSOR (object);

  /* Properties only get set on construction */
  switch (prop_id)
    {
      case PROP_DIMS:
        call_and_warn_about_gerror ("set property 'dimensions'",
                                    torch_tensor_set_dims,
                                    tensor,
                                    static_cast <GList *> (g_value_get_boxed (value)));
        break;
      case PROP_DATA:
        call_and_warn_about_gerror ("set 'data' property",
                                    torch_tensor_set_data,
                                    tensor,
                                    g_value_get_variant (value));
        break;
      default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
        break;
    }
}

/* XXX: There is still no G_TYPE_LIST? */
#define G_TYPE_LIST (g_list_get_type())
static GType
g_list_get_type (void)
{
  static GType type = 0;

  if (type == 0)
    {
      type = g_boxed_type_register_static ("GList",
                                           (GBoxedCopyFunc) g_list_copy, (GBoxedFreeFunc) g_list_free);
    }

  return type;
}


static void
torch_tensor_class_init (TorchTensorClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->constructed = torch_tensor_constructed;
  object_class->get_property = torch_tensor_get_property;
  object_class->set_property = torch_tensor_set_property;
  object_class->finalize = torch_tensor_finalize;

  /**
   * TorchTensor:dimensions: (type GLib.List(guint)) (transfer full)
   *
   * The dimensions of the tensor.
   *
   * Arrays can be N-dimensional, as indicated by the number of
   * elements in the array. For instance, a Tensor with dimension
   * [3, 4, 5] has 3 rows, 4 columns and 5 stacks.
   */
  torch_tensor_props[PROP_DIMS] = g_param_spec_boxed ("dimensions",
                                                      "Dimensions",
                                                      "Dimensions of the Tensor",
                                                      G_TYPE_LIST,
                                                      static_cast <GParamFlags> (G_PARAM_READWRITE |
                                                                                 G_PARAM_CONSTRUCT));

  /**
   * TorchTensor:data:
   *
   * The data of the tensor as a nested array of arrays of variants,
   * with the leaf variants being arrays of concrete types.
   *
   * When set, the tensor will be automatically resized and adopt
   * the dimensionality of the nested array of variants. It
   * is the programmer's responsibility to ensure that
   * sub-array sizes are consistent between sub-arrays
   * of the same dimension and that the underlying datatype
   * is consistent between all sub-arrays. Error cannot be thrown
   * from accessing properties, so if an error occurs %NULL will
   * be returned and an error message printed to the standard out. If
   * you need to handle errors, use %torch_tensor_set_data instead.
   *
   * PyTorch will likely copy the contents of the array
   * either into CPU memory or GPU memory as a result of
   * calling this function, so this property should be used
   * seldomly.
   */
  torch_tensor_props[PROP_DATA] = g_param_spec_variant ("data",
                                                        "Data",
                                                        "Data of the Tensor",
                                                        G_VARIANT_TYPE ("av"),
                                                        nullptr,
                                                        static_cast <GParamFlags> (G_PARAM_READWRITE));

  /**
   * TorchTensor:dtype:
   *
   * The data type of the tensor.
   */
  torch_tensor_props[PROP_DTYPE] = g_param_spec_gtype ("dtype",
                                                       "DType",
                                                       "Data Type",
                                                       G_TYPE_NONE,
                                                       static_cast <GParamFlags> (G_PARAM_READABLE));

  g_object_class_install_properties (object_class,
                                     NPROPS,
                                     torch_tensor_props);
}

TorchTensor *
torch_tensor_new (void)
{
  return static_cast <TorchTensor *> (g_object_new (TORCH_TYPE_TENSOR, NULL));
}

TorchTensor *
torch_tensor_new_from_data (GVariant *data)
{
  return static_cast <TorchTensor *> (g_object_new (TORCH_TYPE_TENSOR, "data", data, NULL));
}

TorchTensor *
torch_tensor_new_from_real_tensor (torch::Tensor const &real_tensor)
{
  g_autoptr (TorchTensor) tensor = static_cast <TorchTensor *> (g_object_new (TORCH_TYPE_TENSOR, NULL));
  TorchTensorPrivate *priv = TORCH_TENSOR_GET_PRIVATE (tensor);

  delete priv->internal;
  priv->internal = new torch::Tensor (real_tensor);

  return static_cast <TorchTensor *> (g_steal_pointer (&tensor));
}


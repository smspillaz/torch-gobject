import argparse
import functools
import json
import os
import re
import sys

from copy import copy

_RE_CAMEL_CASE1 = re.compile(r"([A-Z]+)([A-Z\d][a-z])")
_RE_CAMEL_CASE2 = re.compile(r"([a-z])([A-Z\d])")


def camel_case_to_snake_case(camel_cased):
    camel_cased = _RE_CAMEL_CASE1.sub(r"\1_\2", camel_cased)
    camel_cased = _RE_CAMEL_CASE2.sub(r"\1_\2", camel_cased)
    camel_cased = camel_cased.replace("-", "_")

    return camel_cased.upper()


def indent(text, indent):
    pad = " " * indent
    return "\n".join([pad + line for line in text.splitlines()])

TYPE_MAPPING = {
    "at::ArrayRef<double>": {
        "name": "GArray *",
        "meta": {
            "type": "double"
        },
        "convert_native_qualifiers": "",
        "convert_native_func": lambda a: "torch_array_ref_from_garray <double> ({a})".format(a=a),
        "convert_gobject_prefix": lambda a: "g_autoptr ({a})".format(a=a.strip("* ")),
        "convert_gobject_func": lambda a: "torch_new_garray_from_array_ref <double> ({a})".format(a=a),
    },
    "at::IntArrayRef": {
        "name": "GArray *",
        "meta": {
            "type": "long",
        },
        "convert_native_qualifiers": "",
        "convert_native_func": lambda a: "torch_array_ref_from_garray <long> ({a})".format(a=a),
        "convert_gobject_prefix": lambda a: "g_autoptr ({a})".format(a=a.strip("* ")),
        "convert_gobject_func": lambda a: "torch_new_garray_from_array_ref <long> ({a})".format(a=a),
    },
    "at::Device": {
        "name": "TorchDevice *",
        "convert_native_qualifiers": "&",
        "convert_native_func": lambda a: "torch_convert_to_real ({a})".format(a=a),
        "convert_gobject_prefix": lambda a: "g_autoptr ({a})".format(a=a.strip("* ")),
        "convert_gobject_func": lambda a: "torch_convert_to_gobject ({a})".format(a=a),
    },
    "at::Dimname": {
        "name": "TorchDimname *",
        "convert_native_qualifiers": "&",
        "convert_native_func": lambda a: "torch_convert_to_real ({a})".format(a=a),
        "convert_gobject_prefix": lambda a: "g_autoptr ({a})".format(a=a.strip("* ")),
        "convert_gobject_func": lambda a: "torch_convert_to_gobject ({a})".format(a=a),
    },
    "at::DimnameList": {
        "name": "GPtrArray *",
        "meta": {
            "type": "TorchDimname *"
        },
        "convert_native_qualifiers": "",
        "convert_native_func": lambda a: "torch_dimname_list_from_dimname_ptr_array ({a})".format(a=a),
        "convert_gobject_prefix": lambda a: "g_autoptr ({a})".format(a=a.strip("* ")),
        "convert_gobject_func": lambda a: "torch_tensor_ptr_array_from_dimname_list ({a})".format(a=a),
    },
    "at::Generator": {
        "name": "TorchGenerator *",
        "convert_native_qualifiers": "&",
        "convert_native_func": lambda a: "torch_convert_to_real ({a})".format(a=a),
        "convert_gobject_prefix": lambda a: "g_autoptr ({a})".format(a=a.strip("* ")),
        "convert_gobject_func": lambda a: "torch_convert_to_gobject ({a})".format(a=a),
    },
    "at::MemoryFormat": {
        "name": "TorchMemoryFormat",
        "convert_native_qualifiers": "",
        "convert_native_func": lambda a: "torch_convert_to_real ({a})".format(a=a),
        "convert_gobject_prefix": lambda a: a,
        "convert_gobject_func": lambda a: "torch_convert_to_gobject ({a})".format(a=a),
    },
    "at::Scalar": {
        "name": "GValue *",
        "convert_native_qualifiers": "",
        "convert_native_func": lambda a: "torch_scalar_from_gvalue ({a})".format(a=a),
        "convert_gobject_prefix": lambda a: "g_autofree {a}".format(a=a),
        "convert_gobject_func": lambda a: "torch_gvalue_from_scalar ({a})".format(a=a),
    },
    "at::ScalarType": {
        "name": "GType",
        "convert_native_qualifiers": "",
        "convert_native_func": lambda a: "torch_scalar_type_from_gtype ({a})".format(a=a),
        "convert_gobject_prefix": lambda a: a,
        "convert_gobject_func": lambda a: "torch_gtype_from_scalar_type ({a})".format(a=a),
    },
    "at::Storage": {
        "name": "TorchStorage *",
        "convert_native_qualifiers": "&",
        "convert_native_func": lambda a: "torch_convert_to_real ({a})".format(a=a),
        "convert_gobject_prefix": lambda a: "g_autoptr ({a})".format(a=a.strip("* ")),
        "convert_gobject_func": lambda a: "torch_convert_to_gobject ({a})".format(a=a),
    },
    "at::Tensor": {
        "name": "TorchTensor *",
        "convert_native_qualifiers": "&",
        "convert_native_func": lambda a: "torch_convert_to_real ({a})".format(a=a),
        "convert_gobject_prefix": lambda a: "g_autoptr ({a})".format(a=a.strip("* ")),
        "convert_gobject_func": lambda a: "torch_convert_to_gobject ({a})".format(a=a),
    },
    "at::TensorList": {
        "name": "GPtrArray *",
        "meta": {
            "type": "TorchTensor *"
        },
        "convert_native_qualifiers": "",
        "convert_native_func": lambda a: "torch_tensor_list_from_tensor_ptr_array ({a})".format(a=a),
        "convert_gobject_prefix": lambda a: "g_autoptr ({a})".format(a=a.strip("* ")),
        "convert_gobject_func": lambda a: "torch_tensor_ptr_array_from_tensor_list ({a})".format(a=a),
    },
    "at::TensorOptions": {
        "name": "TorchTensorOptions *",
        "convert_native_qualifiers": "&",
        "convert_native_func": lambda a: "torch_convert_to_real ({a})".format(a=a),
        "convert_gobject_prefix": lambda a: "g_autoptr ({a})".format(a=a.strip("* ")),
        "convert_gobject_func": lambda a: "torch_convert_to_gobject ({a})".format(a=a),
    },
    "bool": {
        "name": "gboolean",
        "convert_native_qualifiers": "",
        "convert_native_func": lambda a: a,
        "convert_gobject_prefix": lambda a: a,
        "convert_gobject_func": lambda a: a,
    },
    "double": {
        "name": "double",
        "convert_native_qualifiers": "",
        "convert_native_func": lambda a: a,
        "convert_gobject_prefix": lambda a: a,
        "convert_gobject_func": lambda a: a,
    },
    "int64_t": {
        "name": "int64_t",
        "convert_native_qualifiers": "",
        "convert_native_func": lambda a: a,
        "convert_gobject_prefix": lambda a: a,
        "convert_gobject_func": lambda a: a,
    },
    "long": {
        "name": "long",
        "convert_native_qualifiers": "",
        "convert_native_func": lambda a: a,
        "convert_gobject_prefix": lambda a: a,
        "convert_gobject_func": lambda a: a,
    },
    "std::string": {
        "name": "const char *",
        "convert_native_qualifiers": "",
        "convert_native_func": lambda a: "std::string ({a})".format(a=a),
        "convert_gobject_prefix": lambda a: "g_autofree {a}".format(a=a),
        "convert_gobject_func": lambda a: "g_strdup ({a}.c_str())".format(a=a),
    },
    "c10::string_view": {
        "name": "const char *",
        "convert_native_qualifiers": "",
        "convert_native_func": lambda a: "c10::string_view ({a})".format(a=a),
        "convert_gobject_prefix": lambda a: "g_autofree {a}".format(a=a),
        "convert_gobject_func": lambda a: "g_strdup ({a}.data())".format(a=a),
    },
    "c10::List<c10::optional<at::Tensor>>": {
        "name": "GPtrArray *",
        "meta": {
            "type": "TorchTensor *",
            "nullable_elements": True,
        },
        "convert_native_qualifiers": "",
        "convert_native_func": lambda a: "torch_optional_tensor_list_from_tensor_ptr_array ({a})".format(a=a),
        "convert_gobject_prefix": lambda a: "g_autoptr ({a})".format(a=a.strip("* ")),
        "convert_gobject_func": lambda a: "torch_tensor_ptr_array_from_optional_tensor_list ({a})".format(a=a),
    }
}

CONVERSIONS = {
    "GArray *": lambda name, meta: f"torch_array_ref_from_garray <{meta['convert_type']}> ({name})",
    "GPtrArray *": lambda name, meta: f"torch_list_from_gptrarray <{meta['type']}> ({name})",
    "TorchDevice *": lambda name, meta: f"torch_convert_to_real ({name})",
    "TorchTensor *": lambda name, meta: f"torch_convert_to_real ({name})",
    "TorchOptionalValue *": lambda name, meta: f"torch_optional_value_to_c10_optional ({name}, torch_optional_value_get_{meta['type'].lower()})",
    "TorchNNConvPaddingOptions1D *": lambda name, meta: f"torch_convert_to_real ({name})",
    "TorchNNConvPaddingOptions2D *": lambda name, meta: f"torch_convert_to_real ({name})",
    "TorchNNConvPaddingOptions3D *": lambda name, meta: f"torch_convert_to_real ({name})",
    "TorchNNConvPaddingMode": lambda name, meta: f"torch_convert_to_real ({name})",
    "TorchNNEmbeddingBagMode": lambda name, meta: f"torch_convert_to_real ({name})",
    "TorchNNGridSampleMode": lambda name, meta: f"torch_convert_to_real ({name})",
    "TorchNNGridSamplePaddingMode": lambda name, meta: f"torch_convert_to_real ({name})",
    "TorchNNInterpolateMode": lambda name, meta: f"torch_convert_to_real ({name})",
    "TorchNNLossReductionMode": lambda name, meta: f"torch_convert_to_real ({name})",
    "TorchNNAnyModuleCastable *": lambda name, meta: f"torch_nn_any_module_castable_to_real_any_module ({name})",
    "TorchNNPadMode": lambda name, meta: f"torch_convert_to_real ({name})",
    "TorchNNRNNNonlinearityType": lambda name, meta: f"torch_convert_to_real ({name})",
    "TorchNNNamedshapeType": lambda name, meta: f"torch_nn_namedshape_array_to_real_namedshape ({name})",
    "TorchNNTransformerDecoderLayer *": lambda name, meta: f"torch_convert_to_real ({name})",
    "TorchNNTransformerEncoderLayer *": lambda name, meta: f"torch_convert_to_real ({name})",
    "TorchNNTransformerActivationType": lambda name, meta: f"torch_convert_to_real ({name})",
    "TorchNNUpsampleMode": lambda name, meta: f"torch_convert_to_real ({name})",
}

STORAGE = {
    "int64_t *": {
        "container": "GArray *",
        "element_type": "int64_t",
        "convert_func": "torch_new_g_array_from_c_array ({name}, {meta[length]})",
        "copy_func": "g_array_copy ({name})",
    },
    "double *": {
        "container": "GArray *",
        "element_type": "double",
        "convert_func": "torch_new_g_array_from_c_array ({name}, {meta[length]})",
        "copy_func": "g_array_copy ({name})",
    },
    "TorchNNNamedshapeElement *": {
        "container": "GArray *",
        "element_type": "TorchNNNamedshapeElement",
        "convert_func": "torch_new_g_array_from_c_array ({name}, {meta[length]})",
        "copy_func": "g_array_copy ({name})",
    },
}
GOBJECT_CONVERSIONS = {
    "torch::Tensor const &": lambda name, meta: f"torch_tensor_new_from_real_tensor ({name})"
}

ACCESS_FUNCS = {
    "GArray *": "&(g_array_index ({name}, {element_type}, 0))",
    "GPtrArray *": "static_cast <{element_type} *> ({name}->pdata)",
    "TorchCallbackData *": "{name}->callback",
}
ACCESS_LENGTH_FUNCS = {"GArray *": "{name}->len", "GPtrArray *": "{name}->len"}
C_TYPE_TO_INTROSPECTION_TYPE = {
    "double": "gdouble",
    "int64_t": "gint64",
    "TorchNNNamedshapeElement": "TorchNNNamedshapeElement",
}

COPY_G_OBJECT_REF = "g_object_ref"
COPY_TORCH_OPTIONAL_VALUE_COPY = "torch_optional_value_copy"
COPY_FUNCS = {
    "GArray *": "g_array_ref",
    "TorchTensor *": COPY_G_OBJECT_REF,
    "TorchOptionalValue *": COPY_TORCH_OPTIONAL_VALUE_COPY,
    "TorchNNConvPaddingOptions *": "torch_nn_conv_padding_options_copy",
}

DESTROY_G_OBJECT_UNREF = "g_object_unref"
DESTROY_TORCH_OPTIONAL_VALUE_FREE = "torch_optional_value_free"
DESTROY_FUNCS = {
    "GArray *": "g_array_unref",
    "TorchTensor *": DESTROY_G_OBJECT_UNREF,
    "TorchOptionalValue *": DESTROY_TORCH_OPTIONAL_VALUE_FREE,
    "TorchNNConvPaddingOptions *": "torch_nn_conv_padding_options_free",
    "TorchCallbackData *": "torch_callback_data_unref",
}
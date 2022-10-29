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


CONVERSIONS = {
    "GArray *": lambda name, meta: f"torch_array_ref_from_garray <{meta['type']}> ({name})",
    "TorchTensor *": lambda name, meta: f"torch_tensor_get_real_tensor ({name})",
    "TorchOptionalValue *": lambda name, meta: f"torch_optional_value_to_c10_optional ({name}, torch_optional_value_get_{meta['type'].lower()})",
    "TorchNNConvPaddingOptions *": lambda name, meta: f"torch_nn_conv_padding_options_to_real_padding_t <{meta['dims']}> ({name})",
    "TorchNNConvPaddingMode": lambda name, meta: f"torch_nn_conv_padding_mode_to_real_conv_padding_mode ({name})",
    "TorchNNEmbeddingBagMode": lambda name, meta: f"torch_nn_embedding_bag_mode_to_real_embedding_bag_mode ({name})",
    "TorchNNGridSampleMode": lambda name, meta: f"torch_nn_grid_sample_mode_to_real_grid_sample_mode ({name})",
    "TorchNNGridSamplePaddingMode": lambda name, meta: f"torch_nn_grid_sample_padding_mode_to_real_grid_sample_padding_mode ({name})",
    "TorchNNInterpolateMode": lambda name, meta: f"torch_nn_interpolate_mode_to_real_interpolate_mode ({name})",
    "TorchNNAnyModuleCastable *": lambda name, meta: f"torch_nn_any_module_castable_to_real_any_module ({name})",
    "TorchNNPadMode": lambda name, meta: f"torch_nn_pad_mode_to_real_pad_mode ({name})",
    "TorchNNRNNNonlinearityType": lambda name, meta: f"torch_nn_rnn_nonlinearity_type_to_real_rnn_nonlinearity_type ({name})",
    "TorchNNNamedshapeType": lambda name, meta: f"torch_nn_namedshape_array_to_real_namedshape ({name}, {name}->len)",
    "TorchNNTransformerDecoderLayer *": lambda name, meta: f"torch_nn_transformer_decoder_layer_to_real_transformer_decoder_layer ({name})",
    "TorchNNTransformerEncoderLayer *": lambda name, meta: f"torch_nn_transformer_encoder_layer_to_real_transformer_encoder_layer ({name})",
    "TorchNNUpsampleMode": lambda name, meta: f"torch_nn_upsample_mode_to_real_upsample_mode ({name})",
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


def convert_cpp_to_c(opt_info, name):
    wrapped_conversion = GOBJECT_CONVERSIONS.get(
        opt_info["cpp_type"], lambda n, m: name
    )(name, opt_info.get("meta", {}))
    return wrapped_conversion


def convert_callback_to_cpp(opt_info, name):
    callback_c_type = opt_info["c_type"]
    callback_temporary_variable = f"{opt_info['name']}_callback_data_callable_wrapper"
    callback_lambda_wrapper_variable = f"{opt_info['name']}_callback_lambda"
    callback_temporary_capture = f"{callback_temporary_variable} = TorchCallbackDataCallableWrapper<{callback_c_type}> ({name})"
    opt_info_args = opt_info.get("meta", {}).get("args", [])
    opt_rv_c_type = opt_info.get("meta", {}).get("return", {}).get("c_type", "void")
    opt_rv_cpp_type = opt_info.get("meta", {}).get("return", {}).get("cpp_type", "void")
    lambda_decl = (
        f"[{callback_temporary_capture}]("
        + ", ".join([f"{oa['cpp_type']} {oa['name']}" for oa in opt_info_args])
        + ") -> "
        + opt_rv_cpp_type
    )
    convert_arg_lines = [
        f"{oa['c_type']} c_{oa['name']} = {convert_cpp_to_c(oa, oa['name'])};"
        for oa in opt_info_args
    ]
    call_func_line_args = ", ".join([f"c_{oa['name']}" for oa in opt_info_args])
    rv_storage_type = (
        f"g_autoptr ({opt_rv_c_type.strip('*').strip()})"
        if "*" in opt_rv_c_type
        else opt_rv_c_type
    )
    call_func_part = f"{callback_temporary_variable} ({call_func_line_args})"
    call_func_line = (
        f"{rv_storage_type} rv = {call_func_part};"
        if rv_storage_type != "void"
        else f"{call_func_part};"
    )
    convert_rv_part = (
        convert_c_to_cpp(opt_info["meta"]["return"], "rv")
        if rv_storage_type != "void"
        else ""
    )
    return_line = (
        f"return {convert_rv_part};" if rv_storage_type != "void" else "return;"
    )

    return "\n".join(
        [
            lambda_decl,
            "{",
            indent("\n".join(convert_arg_lines + [call_func_line, return_line]), 2),
            "}",
        ]
    )


def convert_c_to_cpp(opt_info, name):
    if opt_info.get("meta", {}).get("func_data_ptr"):
        return convert_callback_to_cpp(opt_info, name)

    storage_type = (
        STORAGE[opt_info["c_type"]]["container"]
        if opt_info["c_type"] in STORAGE
        else opt_info["c_type"]
    )
    meta = (
        {
            **opt_info.get("meta", {}),
            "type": STORAGE[opt_info["c_type"]]["element_type"],
            "length": ACCESS_LENGTH_FUNCS[
                STORAGE[opt_info["c_type"]]["container"]
            ].format(name=name),
        }
        if opt_info["c_type"] in STORAGE
        else opt_info.get("meta", {})
    )

    if "convert_pipeline" in meta:
        return functools.reduce(
            lambda prev, next: next.format(name=prev, meta=meta),
            meta["convert_pipeline"],
            name,
        )

    wrapped_conversion = CONVERSIONS.get(storage_type, lambda n, m: name)(name, meta)
    return wrapped_conversion


def print_opt_struct_source(opt_struct):
    snake_name = camel_case_to_snake_case(opt_struct["name"]).lower()
    convert_func_name = f"torch_{snake_name}_struct_to_options"
    struct_name = f"Torch{opt_struct['name']}"

    print("")
    print(f"{opt_struct['cpp']} {convert_func_name} ({struct_name} *opts)")
    print("{")

    opts_by_name = {opt_info["name"]: opt_info for opt_info in opt_struct["opts"]}
    cpp_args = (
        ""
        if "cpp_constructor" not in opt_struct
        else ", ".join(
            [
                convert_c_to_cpp(opts_by_name[opt_name], f"opts->{opt_name}")
                for opt_name in opt_struct["cpp_constructor"]["args"]
            ]
        )
    )
    skip_args = set(
        []
        if "cpp_constructor" not in opt_struct
        else opt_struct["cpp_constructor"]["args"]
    )

    print(indent(f"auto options = {opt_struct['cpp']}({cpp_args});", 2))
    for opt_info in opt_struct["opts"]:
        if opt_info["name"] in skip_args:
            continue

        wrapped_arg = convert_c_to_cpp(opt_info, f"opts->{opt_info['name']}")

        print("")
        if "*" in opt_info["c_type"]:
            print(indent(f"if (opts->{opt_info['name']} != nullptr)", 2))
            print(indent("{", 4))
            print(indent(f"options = options.{opt_info['name']}({wrapped_arg});", 6))
            print(indent("}", 4))
        else:
            print(indent(f"options = options.{opt_info['name']}({wrapped_arg});", 2))

    print("")
    print(indent("return options;", 2))
    print("}")


def maybe_yield_opt_to_array_constructor_arg(meta):
    length = meta.get("length", None)

    if length is not None:
        try:
            int(length)
        except ValueError:
            yield ("size_t", length)


def maybe_yield_opt_to_callback_constructor_arg(meta):
    func_data_ptr = meta.get("func_data_ptr", None)

    if func_data_ptr is not None:
        yield ("gpointer", func_data_ptr)

    func_data_ptr_destroy = meta.get("func_data_ptr_destroy", None)

    if func_data_ptr_destroy is not None:
        yield ("GDestroyNotify", func_data_ptr_destroy)


def opts_to_constructor_args(opts):
    for opt_info in opts:
        yield (opt_info["c_type"], opt_info["name"])

        meta = opt_info.get("meta", {})
        yield from maybe_yield_opt_to_array_constructor_arg(meta)
        yield from maybe_yield_opt_to_callback_constructor_arg(meta)


def make_array_struct_member_annotation(opt_info):
    storage_type = (
        STORAGE[opt_info["c_type"]]["container"]
        if opt_info["c_type"] in STORAGE
        else opt_info["c_type"]
    )
    storage_element_type = (
        f" (element-type {C_TYPE_TO_INTROSPECTION_TYPE[STORAGE[opt_info['c_type']]['element_type']]})"
        if opt_info["c_type"] in STORAGE
        else ""
    )

    return storage_element_type


def format_struct_member_annotation(opt_info):
    transfer = " (transfer none)" if "*" in opt_info["c_type"] else ""
    array_element_type = make_array_struct_member_annotation(opt_info)
    nullable = " (nullable)" if "*" in opt_info["c_type"] else ""

    annotations = (
        f"{transfer}{array_element_type}{nullable}: "
        if (transfer or array_element_type)
        else " "
    )

    return "@{name}:{annotations}A #{c_type}".format(
        name=opt_info["name"], c_type=opt_info["c_type"], annotations=annotations
    )


def print_opt_struct_header(opt_struct):
    struct_name = f"Torch{opt_struct['name']}"
    snake_name = camel_case_to_snake_case(struct_name).lower()
    constructor = f"{snake_name}_new"
    destructor = f"{snake_name}_free"
    copy = f"{snake_name}_copy"

    print("")
    print("/**")
    print(
        " * "
        + (
            "\n * ".join(
                [f"{struct_name}:"]
                + [
                    format_struct_member_annotation(opt_info)
                    for opt_info in opt_struct["opts"]
                ]
            )
        )
    )
    print(" */")
    print("typedef struct {")
    for opt_info in opt_struct["opts"]:
        storage_info = STORAGE.get(opt_info["c_type"])
        storage_c_type = (
            storage_info["container"]
            if opt_info["c_type"] in STORAGE
            else opt_info["c_type"]
        )

        if "func_data_ptr" in opt_info.get("meta", {}):
            # This is a closure, we need to create a TorchCallbackData
            # to store the callback pointer, func_data_ptr and func_data_destroy_ptr
            storage_c_type = "TorchCallbackData *"
        print(indent(f"{storage_c_type} {opt_info['name']};", 2))
    print(f"}} {struct_name};")
    print("")

    formatted_args = ", ".join(
        # type and name
        list(
            map(
                lambda x: f"{x[0]} {x[1]}", opts_to_constructor_args(opt_struct["opts"])
            )
        )
    )
    print(f"{struct_name} * {constructor} ({formatted_args});")
    print(f"{struct_name} * {copy} ({struct_name} *opts);")
    print(f"void {destructor} ({struct_name} *opts);")
    print("")

    snake_upper_components = snake_name.upper().split("_")
    gobject_type = f"TORCH_TYPE_{'_'.join(snake_upper_components[1:])}"
    print(f"GType {snake_name}_get_type (void);")
    print(f"#define {gobject_type} ({snake_name}_get_type ())")


def print_header(opts):
    for opt_struct in opts:
        print_opt_struct_header(opt_struct)


def generate_header(options):
    print("#include <torch-gobject/torch-optional-value.h>")
    print("#include <torch-gobject/torch-tensor.h>")
    print("#include <torch-gobject/nn/options/torch-nn-conv-padding-options.h>")
    print("#include <torch-gobject/nn/options/torch-nn-embedding-bag-mode.h>")
    print("#include <torch-gobject/nn/options/torch-nn-grid-sample-mode.h>")
    print("#include <torch-gobject/nn/options/torch-nn-grid-sample-padding-mode.h>")
    print('#include "torch-enums.h"')
    print("")
    print("G_BEGIN_DECLS")

    print_header(options)

    print("")
    print("G_END_DECLS")


def print_source(opts):
    for opt_struct in opts:
        print_opt_struct_source(opt_struct)


def generate_source(options):
    print("#include <gio/gio.h>")
    print("#include <torch-gobject/torch-tensor.h>")
    print("#include <torch-gobject/torch-tensor-internal.h>")
    print("#include <torch-gobject/torch-optional-value.h>")
    print("#include <torch-gobject/torch-util.h>")
    print("#include <torch-gobject/nn/options/torch-nn-options-generated.h>")
    print(
        "#include <torch-gobject/nn/options/torch-nn-conv-padding-options-internal.h>"
    )
    print("#include <torch-gobject/nn/options/torch-nn-embedding-bag-mode-internal.h>")
    print("#include <torch-gobject/nn/options/torch-nn-grid-sample-mode-internal.h>")
    print(
        "#include <torch-gobject/nn/options/torch-nn-grid-sample-padding-mode-internal.h>"
    )
    print("")
    print("#include <string>")
    print("#include <vector>")
    print("#include <torch/nn/options.h>")
    print('#include "torch-enums.h"')

    print_source(options)


def access_underlying(variable, opt_info):
    if opt_info["c_type"] in STORAGE:
        storage_info = STORAGE[opt_info["c_type"]]
        return ACCESS_FUNCS.get(storage_info["container"], variable).format(
            name=variable, element_type=storage_info["element_type"]
        )

    return variable


def make_array_annotation(opt_info):
    if "*" not in opt_info["c_type"]:
        return ""

    if opt_info.get("meta", {}).get("length", None) is None:
        return ""

    return f" (array fixed-size={opt_info['meta']['length']})"


def format_arg_annotation(opt_info):
    transfer = " (transfer none)" if "*" in opt_info["c_type"] else ""
    array_length = make_array_annotation(opt_info)
    nullable = " (nullable)" if "*" in opt_info["c_type"] else ""

    annotations = (
        f"{transfer}{array_length}{nullable}: " if (transfer or array_length) else " "
    )

    return "@{name}:{annotations}A #{c_type}".format(
        name=opt_info["name"], c_type=opt_info["c_type"], annotations=annotations
    )


def print_opt_struct_introspectable_source(opt_struct):
    struct_name = f"Torch{opt_struct['name']}"
    snake_name = camel_case_to_snake_case(struct_name).lower()
    constructor = f"{snake_name}_new"
    destructor = f"{snake_name}_free"
    copy = f"{snake_name}_copy"

    print("")

    formatted_args = ", ".join(
        [f"{opt_info['c_type']} {opt_info['name']}" for opt_info in opt_struct["opts"]]
    )
    print("/**")
    print(
        " * "
        + (
            "\n * ".join(
                [f"{constructor}:"]
                + [format_arg_annotation(opt_info) for opt_info in opt_struct["opts"]]
                + ["", f"Returns: (transfer full): A new #{struct_name}"]
            )
        )
    )
    print(" */")
    print(f"{struct_name} * {constructor} ({formatted_args})")
    print("{")
    print(indent(f"{struct_name} *opts = g_new0({struct_name}, 1);", 2))
    print("")
    for opt_info in opt_struct["opts"]:
        storage_info = STORAGE.get(opt_info["c_type"])

        # We have a custom storage container, so we need to wrap
        # the value into the container first
        if storage_info is not None:
            print(
                indent(
                    f"opts->{opt_info['name']} = {storage_info['convert_func'](opt_info['name'], opt_info.get('meta', {}))};",
                    2,
                )
            )
        elif opt_info["c_type"] in COPY_FUNCS:
            print(
                indent(
                    f"opts->{opt_info['name']} = {opt_info['name']} != NULL ? {COPY_FUNCS[opt_info['c_type']]} ({opt_info['name']}) : NULL;",
                    2,
                )
            )
        else:
            print(indent(f"opts->{opt_info['name']} = {opt_info['name']};", 2))
    print("")
    print(indent("return opts;", 2))
    print("}")
    print("")
    print("/**")
    print(
        " * "
        + (
            "\n * ".join(
                [
                    f"{copy}:",
                    f"@opts: (transfer none): The #{struct_name} to copy.",
                    "",
                    f"Returns: (transfer full): A new #{struct_name} which is a copy of @opts.",
                ]
            )
        )
    )
    print(" */")
    print(f"{struct_name} * {copy} ({struct_name} *opts)")
    print("{")
    print(
        indent(
            f"return {constructor} "
            f"({', '.join([access_underlying('opts->{}'.format(opt_info['name']), opt_info) for opt_info in opt_struct['opts']])});",
            2,
        )
    )
    print("}")
    print("")
    print("/**")
    print(
        " * "
        + (
            "\n * ".join(
                [
                    f"{destructor}:",
                    f"@opts: (transfer none): The #{struct_name} to free.",
                ]
            )
        )
    )
    print(" */")
    print(f"void {destructor} ({struct_name} *opts)")
    print("{")
    for opt_info in opt_struct["opts"]:
        storage_c_type = (
            STORAGE[opt_info["c_type"]]["container"]
            if opt_info["c_type"] in STORAGE
            else opt_info["c_type"]
        )
        if storage_c_type in DESTROY_FUNCS:
            print(
                indent(
                    f"g_clear_pointer (&opts->{opt_info['name']}, {DESTROY_FUNCS[storage_c_type]});",
                    2,
                )
            )
    print(indent("g_clear_pointer ((gpointer **) &opts, g_free);", 2))
    print("}")
    print("")
    print(
        f"G_DEFINE_BOXED_TYPE ({struct_name}, {snake_name}, (GBoxedCopyFunc) {copy}, (GBoxedFreeFunc) {destructor})"
    )


def generate_introspectable_source(options):
    print("#include <torch-gobject/torch-util.h>")
    print("#include <torch-gobject/nn/options/torch-nn-options-generated.h>")
    print("")

    for opt_struct in options:
        print_opt_struct_introspectable_source(opt_struct)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("defs", help="Definitions JSON file to parse")
    parser.add_argument("--output", help="Where to write the file")
    parser.add_argument("--header", action="store_true", help="Writing a header file")
    parser.add_argument("--source", action="store_true", help="Writing a source file")
    parser.add_argument(
        "--introspectable-source",
        action="store_true",
        help="Writing an introspectable source file",
    )
    args = parser.parse_args()

    if args.output:
        sys.stdout = open(args.output, "wt")

    with open(args.defs, "r") as f:
        options = json.load(f)

    if args.header:
        generate_header(options)

    if args.source:
        generate_source(options)

    if args.introspectable_source:
        generate_introspectable_source(options)


if __name__ == "__main__":
    main()

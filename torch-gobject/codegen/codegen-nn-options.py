import argparse
import functools
import json
import os
import re
import sys

from copy import copy

from common import (
    ACCESS_FUNCS,
    ACCESS_LENGTH_FUNCS,
    C_TYPE_TO_INTROSPECTION_TYPE,
    CONVERSIONS,
    COPY_FUNCS,
    DESTROY_FUNCS,
    GOBJECT_CONVERSIONS,
    STORAGE,
    _RE_CAMEL_CASE1,
    _RE_CAMEL_CASE2,
    camel_case_to_snake_case,
    fmt_annotations,
    fmt_function_decl_header_comment,
    fmt_gobject_func_fwd_decl,
    fmt_introspectable_struct_constructor_source,
    fmt_introspectable_struct_copy_source,
    indent,
)


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
        (
            f"g_autoptr ({oa['c_type'].strip('*').strip()})"
            if "*" in oa["c_type"]
            else oa["c_type"]
        )
        + f" c_{oa['name']} = {convert_cpp_to_c(oa, oa['name'])};"
        for oa in opt_info_args
    ]
    call_func_line_args = ", ".join([f"c_{oa['name']}" for oa in opt_info_args])
    call_func_part = f"{callback_temporary_variable} ({call_func_line_args})"
    call_func_line = (
        f"{opt_rv_c_type} rv = {call_func_part};"
        if opt_rv_c_type != "void"
        else f"{call_func_part};"
    )
    convert_rv_part = (
        convert_c_to_cpp(opt_info["meta"]["return"], "rv")
        if opt_rv_c_type != "void"
        else ""
    )
    return_line = f"return {convert_rv_part};" if opt_rv_c_type != "void" else "return;"

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
            "type": STORAGE[opt_info["c_type"]]["element_type"],
            "convert_type": STORAGE[opt_info["c_type"]]["element_type"],
            **opt_info.get("meta", {}),
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
            yield ("size_t", length, {})


def maybe_yield_opt_to_callback_constructor_arg(meta):
    func_data_ptr = meta.get("func_data_ptr", None)

    if func_data_ptr is not None:
        yield ("gpointer", func_data_ptr, {})

    func_data_ptr_destroy = meta.get("func_data_ptr_destroy", None)

    if func_data_ptr_destroy is not None:
        yield ("GDestroyNotify", func_data_ptr_destroy, {})


def opts_to_constructor_args(opts):
    for opt_info in opts:
        yield (opt_info["c_type"], opt_info["name"], opt_info.get("meta", {}))

        meta = opt_info.get("meta", {})
        yield from maybe_yield_opt_to_array_constructor_arg(meta)
        yield from maybe_yield_opt_to_callback_constructor_arg(meta)


def get_array_struct_member_element_type(opt_info):
    storage_type = (
        STORAGE[opt_info["c_type"]]["container"]
        if opt_info["c_type"] in STORAGE
        else opt_info["c_type"]
    )
    storage_element_type = (
        C_TYPE_TO_INTROSPECTION_TYPE[STORAGE[opt_info["c_type"]]["element_type"]]
        if opt_info["c_type"] in STORAGE
        else None
    )

    return storage_element_type


def format_struct_member_annotation(opt_info):
    annotations = fmt_annotations(
        {
            "type": opt_info["c_type"],
            "element-type": get_array_struct_member_element_type(opt_info),
            "transfer": "none" if "*" in opt_info["c_type"] else None,
            "nullable": True if "*" in opt_info["c_type"] else None,
        }
    )

    return "@{name}{annotations}: A #{c_type}".format(
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
    print("#include <torch-gobject/torch-callback-data.h>")
    print("#include <torch-gobject/nn/torch-nn-any-module-castable.h>")
    print("#include <torch-gobject/nn/torch-nn-any-module.h>")
    print("#include <torch-gobject/nn/torch-nn-distance-function.h>")
    print("#include <torch-gobject/nn/torch-nn-transformer-decoder-layer.h>")
    print("#include <torch-gobject/nn/torch-nn-transformer-encoder-layer.h>")
    print("#include <torch-gobject/nn/torch-nn-module-base.h>")
    print("#include <torch-gobject/nn/options/torch-nn-conv-padding-mode.h>")
    print("#include <torch-gobject/nn/options/torch-nn-conv-padding-options.h>")
    print("#include <torch-gobject/nn/options/torch-nn-embedding-bag-mode.h>")
    print("#include <torch-gobject/nn/options/torch-nn-grid-sample-mode.h>")
    print("#include <torch-gobject/nn/options/torch-nn-grid-sample-padding-mode.h>")
    print("#include <torch-gobject/nn/options/torch-nn-interpolate-mode.h>")
    print("#include <torch-gobject/nn/options/torch-nn-loss-reduction-mode.h>")
    print("#include <torch-gobject/nn/options/torch-nn-namedshape-element.h>")
    print("#include <torch-gobject/nn/options/torch-nn-pad-mode.h>")
    print("#include <torch-gobject/nn/options/torch-nn-rnn-nonlinearity-type.h>")
    print("#include <torch-gobject/nn/options/torch-nn-transformer-activation-type.h>")
    print("#include <torch-gobject/nn/options/torch-nn-upsample-mode.h>")
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
    print("#include <torch-gobject/torch-callback-data-internal.h>")
    print("#include <torch-gobject/torch-tensor.h>")
    print("#include <torch-gobject/torch-tensor-internal.h>")
    print("#include <torch-gobject/torch-optional-value.h>")
    print("#include <torch-gobject/torch-util.h>")
    print("#include <torch-gobject/nn/torch-nn-any-module-internal.h>")
    print("#include <torch-gobject/nn/torch-nn-any-module-castable-internal.h>")
    print("#include <torch-gobject/nn/torch-nn-transformer-decoder-layer-internal.h>")
    print("#include <torch-gobject/nn/torch-nn-transformer-encoder-layer-internal.h>")
    print("#include <torch-gobject/nn/options/torch-nn-options-generated.h>")
    print("#include <torch-gobject/nn/options/torch-nn-conv-padding-mode-internal.h>")
    print(
        "#include <torch-gobject/nn/options/torch-nn-conv-padding-options-internal.h>"
    )
    print("#include <torch-gobject/nn/options/torch-nn-embedding-bag-mode-internal.h>")
    print("#include <torch-gobject/nn/options/torch-nn-grid-sample-mode-internal.h>")
    print(
        "#include <torch-gobject/nn/options/torch-nn-grid-sample-padding-mode-internal.h>"
    )
    print("#include <torch-gobject/nn/options/torch-nn-interpolate-mode-internal.h>")
    print("#include <torch-gobject/nn/options/torch-nn-loss-reduction-mode-internal.h>")
    print("#include <torch-gobject/nn/options/torch-nn-namedshape-element-internal.h>")
    print("#include <torch-gobject/nn/options/torch-nn-pad-mode-internal.h>")
    print(
        "#include <torch-gobject/nn/options/torch-nn-rnn-nonlinearity-type-internal.h>"
    )
    print(
        "#include <torch-gobject/nn/options/torch-nn-transformer-activation-type-internal.h>"
    )
    print("#include <torch-gobject/nn/options/torch-nn-upsample-mode-internal.h>")
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


def opts_to_access_args(opts_struct_name, opts):
    for opt_info in opts:
        opt_name = opt_info["name"]
        struct_member = f"{opts_struct_name}->{opt_name}"

        opt_length = opt_info.get("meta", {}).get("length", None)

        if opt_length is not None:
            try:
                int(opt_length)
            except ValueError:
                if opt_info["c_type"] in STORAGE:
                    storage_container = STORAGE[opt_info["c_type"]]["container"]
                else:
                    storage_container = opt_info["c_type"]

                assert storage_container in ACCESS_LENGTH_FUNCS

                yield ACCESS_LENGTH_FUNCS[storage_container].format(name=struct_member)

        yield access_underlying(struct_member, opt_info)


def print_opt_struct_introspectable_constructor_source(
    constructor, struct_name, opt_struct
):
    print(
        fmt_introspectable_struct_constructor_source(
            constructor, struct_name, {"members": opt_struct["opts"]}
        )
    )


def print_opt_struct_introspectable_destructor_source(
    destructor, struct_name, opt_struct
):
    destructor_return_info = {"type": "void"}
    destructor_args = [
        {
            "name": "opts",
            "type": f"{struct_name} *",
            "transfer": "full",
            "desc": f"The #{struct_name} to free.",
        }
    ]

    print(
        fmt_function_decl_header_comment(
            destructor,
            destructor_return_info,
            destructor_args,
        )
    )
    print(
        fmt_gobject_func_fwd_decl(destructor, destructor_return_info, destructor_args)
    )
    print("{")
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


def print_opt_struct_introspectable_copy_source(copy, struct_name, opt_struct):
    return print(
        fmt_introspectable_struct_copy_source(
            copy,
            struct_name,
            {
                "members": opt_struct["opts"]
            }
        )
    )

    copy_return_info = {
        "type": f"{struct_name} *",
        "transfer": "full",
        "desc": f"A new #{struct_name} which is a copy of @opts",
    }
    copy_args = [
        {
            "name": "opts",
            "type": f"{struct_name} *",
            "transfer": "full",
            "desc": f"The #{struct_name} to copy from.",
        }
    ]

    print(
        fmt_function_decl_header_comment(
            copy,
            copy_return_info,
            copy_args,
        )
    )
    print(
        fmt_gobject_func_fwd_decl(
            copy,
            copy_return_info,
            copy_args,
        )
    )

    print("{")
    print(indent(f"{struct_name} *new_opts = g_new0({struct_name}, 1);", 2))
    # Here we have to be a bit more careful than just re-calling the constructor
    # since we need to take care of closure arguments and copying arrays etc
    for opt_info in opt_struct["opts"]:
        storage_info = STORAGE.get(opt_info["c_type"])

        if storage_info is None:
            if "func_data_ptr" in opt_info.get("meta", {}):
                # This is a closure, we need to create a TorchCallbackData
                # to store the callback pointer, func_data_ptr and func_data_destroy_ptr
                storage_info = {
                    "copy_func": f"torch_callback_data_ref (opts->{opt_info['name']});"
                }

        # We have a custom storage container, so we need to wrap
        # the value into the container first
        if storage_info is not None:
            print(
                indent(
                    f"new_opts->{opt_info['name']} = {storage_info['copy_func'].format(name='opts->' + opt_info['name'])};",
                    2,
                )
            )
        elif opt_info["c_type"] in COPY_FUNCS:
            print(
                indent(
                    f"new_opts->{opt_info['name']} = opts->{opt_info['name']} != NULL ? {COPY_FUNCS[opt_info['c_type']]} (opts->{opt_info['name']}) : NULL;",
                    2,
                )
            )
        else:
            print(
                indent(f"new_opts->{opt_info['name']} = opts->{opt_info['name']};", 2)
            )
    print(indent("return new_opts;", 2))
    print("}")
    print("")


def print_opt_struct_introspectable_source(opt_struct):
    struct_name = f"Torch{opt_struct['name']}"
    snake_name = camel_case_to_snake_case(struct_name).lower()
    constructor = f"{snake_name}_new"
    destructor = f"{snake_name}_free"
    copy = f"{snake_name}_copy"

    print_opt_struct_introspectable_constructor_source(
        constructor, struct_name, opt_struct
    )
    print_opt_struct_introspectable_copy_source(copy, struct_name, opt_struct)
    print_opt_struct_introspectable_destructor_source(
        destructor, struct_name, opt_struct
    )

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

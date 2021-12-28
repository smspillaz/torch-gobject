import argparse
import json
import os
import re
import sys

from copy import copy

_RE_CAMEL_CASE1 = re.compile(r"(.)([A-Z][a-z]+)")
_RE_CAMEL_CASE2 = re.compile(r"([a-z0-9])([A-Z])")


def camel_case_to_snake_case(camel_cased):
    camel_cased = _RE_CAMEL_CASE1.sub(r"\1_\2", camel_cased)
    return _RE_CAMEL_CASE2.sub(r"\1_\2", camel_cased).upper()


def indent(text, indent):
    pad = " " * indent
    return "\n".join([pad + line for line in text.splitlines()])


CONVERSIONS = {
    "TorchTensor *": lambda name, meta: "torch_tensor_get_real_tensor ({name})".format(
        name=name
    ),
    "TorchOptionalValue *": lambda name, meta: "torch_optional_value_to_c10_optional ({name}, torch_optional_value_get_{type})".format(
        type=meta["type"], name=name
    ),
}

COPY_G_OBJECT_REF = "g_object_ref"
COPY_TORCH_OPTIONAL_VALUE_COPY = "torch_optional_value_copy"
COPY_FUNCS = {
    "TorchTensor *": COPY_G_OBJECT_REF,
    "TorchOptionalValue *": COPY_TORCH_OPTIONAL_VALUE_COPY,
}

DESTROY_G_OBJECT_UNREF = "g_object_unref"
DESTROY_TORCH_OPTIONAL_VALUE_FREE = "torch_optional_value_free"
DESTROY_FUNCS = {
    "TorchTensor *": DESTROY_G_OBJECT_UNREF,
    "TorchOptionalValue *": DESTROY_TORCH_OPTIONAL_VALUE_FREE,
}


def convert_c_to_cpp(opt_info, name):
    return CONVERSIONS.get(opt_info["c_type"], lambda name, meta: name)(
        name, opt_info.get("meta", {})
    )


def print_opt_struct_source(opt_struct):
    snake_name = camel_case_to_snake_case(opt_struct["name"]).lower()
    convert_func_name = f"torch_{snake_name}_struct_to_options"
    struct_name = f"Torch{opt_struct['name']}Options"

    print("")
    print(f"{opt_struct['cpp']} {convert_func_name} ({struct_name} *opts)")
    print("{")
    print(indent(f"auto options = {opt_struct['cpp']}();", 2))
    for opt_info in opt_struct["opts"]:
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


def print_opt_struct_header(opt_struct):
    struct_name = f"Torch{opt_struct['name']}Options"
    snake_name = camel_case_to_snake_case(struct_name).lower()
    constructor = f"{snake_name}_new"
    destructor = f"{snake_name}_free"
    copy = f"{snake_name}_copy"

    print("")
    print("typedef struct {")
    for opt_info in opt_struct["opts"]:
        print(indent(f"{opt_info['c_type']} {opt_info['name']};", 2))
    print(f"}} {struct_name};")
    print("")

    formatted_args = ", ".join(
        [f"{opt_info['c_type']} {opt_info['name']}" for opt_info in opt_struct["opts"]]
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
    print("#include <torch-gobject/torch-nn-options-generated.h>")
    print("#include <torch-gobject/torch-optional-value.h>")
    print("#include <torch-gobject/torch-util.h>")
    print("")
    print("#include <string>")
    print("#include <vector>")
    print("#include <torch/nn/options.h>")
    print('#include "torch-enums.h"')

    print_source(options)


def print_opt_struct_introspectable_source(opt_struct):
    struct_name = f"Torch{opt_struct['name']}Options"
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
                + [
                    "@{name}: {transfer}A #{c_type}".format(
                        name=opt_info["name"],
                        c_type=opt_info["c_type"],
                        transfer="(transfer none): "
                        if "*" in opt_info["c_type"]
                        else "",
                    )
                    for opt_info in opt_struct["opts"]
                ]
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
        if opt_info["c_type"] in COPY_FUNCS:
            print(
                indent(
                    f"opts->{opt_info['name']} = {COPY_FUNCS[opt_info['c_type']]} ({opt_info['name']});",
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
            f"return {constructor} ({', '.join(['opts->{}'.format(opt_info['name']) for opt_info in opt_struct['opts']])});",
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
        if opt_info["c_type"] in DESTROY_FUNCS:
            print(
                indent(
                    f"g_clear_pointer (&opts->{opt_info['name']}, {DESTROY_FUNCS[opt_info['c_type']]});",
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
    print("#include <torch-gobject/torch-nn-options-generated.h>")
    print("#include <torch-gobject/torch-util.h>")
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
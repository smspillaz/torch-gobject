import argparse
from collections import defaultdict
from copy import deepcopy
import os
import yaml
import sys

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from common import (
    TYPE_MAPPING,
    fmt_out,
    fmt_transfer,
    fmt_element_type,
    fmt_array_fixed_size,
    fmt_nullable,
    fmt_annotations,
)

RENAME_LIST = {"set_data": "set_data_from_tensor"}


def non_namespaced_function_name(decl):
    inplace = ""
    overload = ""
    overload_name = ""
    op = ""

    if decl["name"].startswith("__"):
        op = "op"

    if decl["name"].endswith("_"):
        inplace = "inplace"

    # Different from overload_name!
    if "overload" in decl:
        overload = decl["overload"].lower()

    if "overload_name" in decl:
        overload_name = decl["overload_name"].lower()

    return "_".join(
        [
            x
            for x in [
                op,
                RENAME_LIST.get(decl["name"], decl["name"]).strip("_"),
                overload_name,
                overload,
                inplace,
            ]
            if x
        ]
    )


def function_name(decl):
    method = ""

    if "Tensor" in decl["method_of"]:
        method = "tensor"

    return "_".join(
        [x for x in ["torch", method, non_namespaced_function_name(decl)] if x]
    )


def unqualified_dynamic_type(type_spec):
    unconst = type_spec.replace("const ", "")
    noqual = unconst.replace("*", "").replace("&", "")
    return noqual.strip()


def map_type_name(type_spec):
    unqualified = unqualified_dynamic_type(type_spec["dynamic_type"])

    if unqualified == "IntArrayRef" and type_spec.get("size", None):
        return "const long *"

    return TYPE_MAPPING[unqualified]["name"]


def map_type_native_conv(type_spec):
    unqualified = unqualified_dynamic_type(type_spec["dynamic_type"])

    if unqualified == "IntArrayRef" and type_spec.get("size", None):
        return lambda a: "torch_array_ref_from_fixed_array({a}, {s})".format(
            a=a, s=type_spec["size"]
        )

    return TYPE_MAPPING[unqualified]["convert_native_func"]


def map_element_type(type_spec):
    unqualified = unqualified_dynamic_type(type_spec["dynamic_type"])

    if unqualified == "IntArrayRef" and type_spec.get("size", None):
        return None

    return TYPE_MAPPING[unqualified].get("meta", {}).get("type", None)


def is_nullable(type_spec):
    unqualified = unqualified_dynamic_type(type_spec["dynamic_type"])

    if TYPE_MAPPING[unqualified].get("meta", {}).get("nullable_elements", False):
        # The container isn't nullable, but the elements may be
        return False

    return type_spec.get("is_nullable", False)


def map_nullable(type_spec):
    return is_nullable(type_spec)


def type_spec_to_gobject_type(type_spec):
    return {
        "name": type_spec["name"],
        "type": map_type_name(type_spec),
        "element-type": map_element_type(type_spec),
        "size": type_spec.get("size", None),
        "nullable": map_nullable(type_spec),
        "transfer": type_spec["transfer"],
    }


def determine_return_transfer_mode(func_decl, return_decl):
    if (
        func_decl["schema_order_arguments"]
        and func_decl["schema_order_arguments"][0]["annotation"] == "a!"
    ):
        assert len(func_decl["returns"]) == 1
        return "self"
    elif "*" not in map_type_name(return_decl):
        return "none"

    return "full"


def make_gobject_decl(decl):
    is_method = "namespace" not in decl["method_of"]
    return_rv_directly = False
    returns_and_gobject_transfers = [
        type_spec_to_gobject_type(
            dict(
                **return_decl,
                transfer=determine_return_transfer_mode(decl, return_decl)
            )
        )
        for return_decl in decl["returns"]
    ]

    gobject_arguments = [
        type_spec_to_gobject_type(dict(**a, transfer="none")) for a in decl["arguments"]
    ]

    # If we return a single pointer-typed value
    # then the return value is that pointer according
    # to whatever transfer rules it is subject to. Otherwise
    # the return value is a gboolean
    if len(decl["returns"]) == 1 and "*" in returns_and_gobject_transfers[0]["type"]:
        return_gobject = returns_and_gobject_transfers[0]
        out_arguments = [
            {
                **returns_and_gobject_transfers[0],
                "out": True,
                "type": "{} *".format(returns_and_gobject_transfers[0]["type"]),
                "nullable": True,
            }
        ]
        return_rv_directly = True
    else:
        return_gobject = {
            "name": "",
            "type": "gboolean",
            "size": None,
            "transfer": "none",
            "element-type": None,
            "nullable": False,
        }
        out_arguments = [
            {
                **out_arg,
                "out": True,
                "type": "{} *".format(out_arg["type"]),
                "nullable": True,
            }
            for out_arg in returns_and_gobject_transfers
        ]

    # Append an error argument to the parameters
    error_argument = {
        "type": "GError **",
        "name": "error",
        "nullable": True,
        "transfer": "full",
        "element-type": None,
        "out": True,
        "size": None,
    }

    return {
        "name": function_name(decl),
        "is_method": is_method,
        "returns": return_gobject,
        "arguments": gobject_arguments,
        "out-arguments": out_arguments,
        "error-argument": error_argument,
        "return-rv-directly": return_rv_directly,
    }


def make_gobject_decl_fwd_decl(decl):
    arg_str_list = []

    for a in decl["arguments"]:
        arg_str_list.append(a["type"] + " " + a["name"])

    if not decl["return-rv-directly"]:
        for a in decl["out-arguments"]:
            arg_str_list.append(a["type"] + " " + a["name"])

    arg_str_list.append(
        decl["error-argument"]["type"] + " " + decl["error-argument"]["name"]
    )

    return "".join(
        [
            decl["returns"]["type"] + " ",
            decl["name"],
            " (",
            ", ".join(arg_str_list),
            ")",
        ]
    )


def make_gobject_decl_header(decl):
    return "\n".join(
        [
            "/**",
            " * " + decl["name"] + ":",
        ]
        + [
            " * @{a[name]}{annotations}: A #{a[type]}".format(
                a=a, annotations=fmt_annotations(a)
            )
            for a in decl["arguments"]
        ]
        + [
            " * @{a[name]}{annotations}: An out-param #{a[type]}".format(
                a=a, annotations=fmt_annotations(a)
            )
            for a in (decl["out-arguments"] if not decl["return-rv-directly"] else [])
        ]
        + [
            " * @{a[name]}{annotations}: An error-out #{a[type]}".format(
                a=decl["error-argument"],
                annotations=fmt_annotations(a=decl["error-argument"]),
            )
        ]
        + [" *"]
        + (
            [
                " * Returns{annotations}: A #{ret[type]}".format(
                    ret=decl["returns"], annotations=fmt_annotations(decl["returns"])
                )
            ]
            if decl["returns"]["type"] != "void"
            else []
        )
        + [" */"]
    )


FUNCTION_BLACKLIST = (
    "data",
    "polygamma",  # declaration in Declarations.yaml is broken
    "special_polygamma",
)


def is_skipped(decl):
    if decl["name"] in FUNCTION_BLACKLIST:
        print("Skipped {decl[name]} - in blacklist".format(decl=decl), file=sys.stderr)
        return True

    if decl["name"] == "count_nonzero" and decl["overload_name"] == "":
        print("Skipped count_nonzero - is ambiguous", file=sys.stderr)
        return True

    # Skip internal funcs
    if decl["name"].startswith("_") and not decl["name"].startswith("__"):
        return True

    # Skip "out" versions
    if decl["name"].endswith("_out"):
        return True

    if decl["returns"]:
        for return_type_info in decl["returns"]:
            if (
                unqualified_dynamic_type(return_type_info["dynamic_type"])
                not in TYPE_MAPPING
            ):
                print(
                    "Skipped",
                    decl["name"],
                    "- returns",
                    return_type_info["dynamic_type"],
                    file=sys.stderr,
                )
                return True

    for a in decl["arguments"]:
        if unqualified_dynamic_type(a["dynamic_type"]) not in TYPE_MAPPING:
            print(
                "Skipped", decl["name"], "- takes", a["dynamic_type"], file=sys.stderr
            )
            return True


def should_rewrite_for_scalar(decl):
    for a in decl["arguments"]:
        unqualified = unqualified_dynamic_type(a["dynamic_type"])
        if unqualified == "at::Scalar":
            return True

    return False


def rewrite_decl_for_scalar(decl, scalar_type):
    d = deepcopy(decl)
    for a in d["arguments"]:
        a["api_dynamic_type"] = a["dynamic_type"]
        if unqualified_dynamic_type(a["dynamic_type"]) == "at::Scalar":
            a["dynamic_type"] = scalar_type

    d["overload"] = scalar_type

    return d


def print_function_decl(decl):
    str_list = []

    # Skip internal funcs
    if is_skipped(decl):
        return

    if should_rewrite_for_scalar(decl):
        for t in ("long", "double", "bool"):
            d = rewrite_decl_for_scalar(decl, t)
            print_function_decl(d)

    gobject_decl = make_gobject_decl(decl)

    str_list.append(make_gobject_decl_fwd_decl(gobject_decl) + ";")

    print("")
    print("\n".join(str_list))


def make_argument_marshaller(argument, gobject_argument):
    arg_type = argument.get("api_dynamic_type", argument["dynamic_type"])
    unqualified_arg_type = unqualified_dynamic_type(arg_type)
    wrapped_type = "c10::optional<{}>".format(unqualified_arg_type) if is_nullable(argument) else unqualified_arg_type
    qualified_type = " ".join([wrapped_type, TYPE_MAPPING[unqualified_arg_type]["convert_native_qualifiers"]])

    return " ".join([
        wrapped_type,
        "real_" + argument["name"],
        "=",
        map_type_native_conv(argument)(argument["name"])
    ]) + ";"


def make_argument_marshallers(arguments, gobject_arguments):
    return "\n".join([
        make_argument_marshaller(argument, gobject_argument)
        for argument, gobject_argument in zip(arguments, gobject_arguments)
    ]);


def determine_real_function_call_return_type(return_decl):
    assert len(return_decl) > 0

    if len(return_decl) == 1:
        return return_decl[0]["type"]

    return "std::tuple <{}>".format(", ".join([
        rd["type"] for rd in return_decl
    ]))


def unpack_real_rv_to_gobject_rv(return_decl,
                                 gobject_return_decl,
                                 real_rv_name,
                                 gobject_rv_name,
                                 out_arg=False):
    return_type = return_decl["dynamic_type"]
    unqualified_return_type = unqualified_dynamic_type(return_type)
    gobject_return_type = gobject_return_decl["type"].removesuffix(" *")

    return " ".join([
        TYPE_MAPPING[unqualified_return_type]["convert_gobject_prefix"](gobject_return_type),
        gobject_rv_name,
        "=",
        TYPE_MAPPING[unqualified_return_type]["convert_gobject_func"](real_rv_name) + ";"
    ])


def determine_return_statement_operand(gobject_return_decl, name):
    gobject_return_type = gobject_return_decl["type"].removesuffix(" *")

    # Cannot be "self", was checked earlier
    if gobject_return_decl["transfer"] == "none":
        return_statement_operand = "{}".format(name)
    elif gobject_return_decl["transfer"] == "full":
        return_statement_operand = "static_cast <{}> (g_steal_pointer (&{}))".format(
            gobject_return_type,
            name
        )
    else:
        assert "Section not reachable" and False

    return return_statement_operand


AT_NAMESPACE_ONLY = {
    "normal": {
        "overloads": [
            "Tensor_float",
            "float_Tensor",
            "Tensor_Tensor"
        ]
    }
}


def determine_namespace(decl):
    if decl["name"] in AT_NAMESPACE_ONLY:
        if decl["overload_name"] in AT_NAMESPACE_ONLY[decl["name"]]["overloads"]:
            return "at"

    return "torch"


def make_function_call(decl, gobject_decl):
    before_block = ""

    if "namespace" in decl["method_of"]:
        call = "{namespace}::{name} ({args});".format(
            namespace=determine_namespace(decl),
            name=decl["name"],
            args=", ".join([
                "real_" + a["name"] for a in decl["arguments"]
            ])
        )
    else:
        # It is a method, use the method-call syntax
        call = "{obj}.{name} ({args});".format(
            obj="real_" + decl["arguments"][0]["name"],
            name=decl["name"],
            args=", ".join([
                "real_" + a["name"] for a in decl["arguments"][1:]
            ])
        )

    # Need to init the tensor first
    if "Tensor" in decl["method_of"]:
        before_block = "\n".join([
            "if (!torch_tensor_init_internal ({}, error))".format(
                gobject_decl["arguments"][0]["name"]
            ),
            "  {",
            "    {} rv = 0;".format(gobject_decl["returns"]["type"]),
            "    return rv;",
            "  }",
        ])

    if decl["returns"]:
        assert len(gobject_decl["out-arguments"]) > 0

        if gobject_decl["returns"]["transfer"] == "self":
            assert len(gobject_decl["out-arguments"]) == 1
            assert (gobject_decl["returns"]["type"] + " *") == gobject_decl["out-arguments"][0]["type"]

            convert_statement = ""
            return_statement = "return {};".format(gobject_decl["arguments"][0]["name"])
        else:
            call = " ".join([
                determine_real_function_call_return_type(decl["returns"]),
                "real_rv",
                "=",
                call
            ])

            # This is a tuple return, so we need to unpack tuple arguments
            if len(gobject_decl["out-arguments"]) > 1:
                convert_statement = "\n".join([
                    unpack_real_rv_to_gobject_rv(return_decl,
                                                 gobject_return_decl,
                                                 "std::get<{}> (real_rv)".format(i),
                                                 "gobject_rv{}".format(i))
                    for i, (return_decl, gobject_return_decl) in enumerate(zip(
                        decl["returns"],
                        gobject_decl["out-arguments"]
                    ))
                ])

                assert not gobject_decl["return-rv-directly"]
            else:
                assert len(gobject_decl["out-arguments"]) == 1
                assert len(decl["returns"]) == 1

                # We convert the real rv directly
                convert_statement = unpack_real_rv_to_gobject_rv(
                    decl["returns"][0],
                    gobject_decl["out-arguments"][0],
                    "real_rv",
                    "gobject_rv0",
                    out_arg=True
                )

                if gobject_decl["return-rv-directly"]:
                    (gobject_decl["returns"]["type"] + " *") == gobject_decl["out-arguments"][0]["type"]
                    return_statement = "return {};".format(
                        determine_return_statement_operand(
                            gobject_decl["out-arguments"][0],
                            "gobject_rv0"
                        )
                    )

            if not gobject_decl["return-rv-directly"]:
                return_assignments = "\n".join([
                    "\n".join([
                        "if ({arg} != NULL)".format(arg=out_arg["name"]),
                        indent("*{arg} = {ro};".format(
                            arg=out_arg["name"],
                            ro=determine_return_statement_operand(
                                out_arg,
                                "gobject_rv{index}".format(index=index)
                            )
                        ), 2)
                    ])
                    for index, out_arg in enumerate(gobject_decl["out-arguments"])
                ])
                return_statement = "\n".join([
                    return_assignments,
                    "return TRUE;"
                ])
    else:
        convert_statement = ""
        return_statement = "return TRUE;"

    call_and_return_try_catch_statement = "\n".join([
        "try",
        "  {",
        indent(make_argument_marshallers(decl["arguments"], gobject_decl["arguments"]), 4),
        "",
        indent("\n".join([
            call,
            convert_statement,
            return_statement
        ]), 4),
        "  }",
        "catch (const std::exception &e)",
        "  {",
        indent("\n".join([
            "g_set_error ({error_param_name}, G_IO_ERROR, G_IO_ERROR_FAILED, \"%s\", e.what ());".format(
                error_param_name=gobject_decl["error-argument"]["name"]
            ),
            "{} rv = 0;".format(gobject_decl["returns"]["type"]),
            "return rv;",
        ]), 4),
        "  }",
    ])

    return "\n".join([s for s in [
        before_block,
        call_and_return_try_catch_statement
    ] if s])


def indent(text, indent):
    pad = " " * indent
    return "\n".join([
        pad + line
        for line in text.splitlines()
    ])

def print_function_body(decl):
    str_list = []

    # Skip internal funcs
    if is_skipped(decl):
        return

    if should_rewrite_for_scalar(decl):
        for t in ("long", "double", "bool"):
            d = rewrite_decl_for_scalar(decl, t)
            print_function_body(d)

    gobject_decl = make_gobject_decl(decl)
    str_list.append(make_gobject_decl_header(gobject_decl))
    str_list.append(make_gobject_decl_fwd_decl(gobject_decl))
    str_list.append("{")
    str_list.append(indent(make_function_call(decl, gobject_decl), 4));
    str_list.append("}")

    print("")
    print("\n".join(str_list))

def print_header(declarations):
    print("#include <torch-gobject/torch-allocator.h>")
    print("#include <torch-gobject/torch-device.h>")
    print("#include <torch-gobject/torch-dimname.h>")
    print("#include <torch-gobject/torch-errors.h>")
    print("#include <torch-gobject/torch-generator.h>")
    print("#include <torch-gobject/torch-storage.h>")
    print("#include <torch-gobject/torch-memory-format.h>")
    print("#include <torch-gobject/torch-tensor.h>")
    print("#include <torch-gobject/torch-tensor-options.h>")
    print("")
    print("G_BEGIN_DECLS")

    for d in declarations:
        print_function_decl(d)

    print("G_END_DECLS")

def print_source(declarations):
    print("#include <torch/torch.h>")
    print("#include <gio/gio.h>")
    print("#include <torch-gobject/torch-allocator-internal.h>")
    print("#include <torch-gobject/torch-device-internal.h>")
    print("#include <torch-gobject/torch-device-type-internal.h>")
    print("#include <torch-gobject/torch-dimname-internal.h>")
    print("#include <torch-gobject/torch-dimname-type-internal.h>")
    print("#include <torch-gobject/torch-generator-internal.h>")
    print("#include <torch-gobject/torch-layout-internal.h>")
    print("#include <torch-gobject/torch-memory-format-internal.h>")
    print("#include <torch-gobject/torch-storage-internal.h>")
    print("#include <torch-gobject/torch-tensor.h>")
    print("#include <torch-gobject/torch-tensor-internal.h>")
    print("#include <torch-gobject/torch-tensor-options-internal.h>")
    print("#include <torch-gobject/torch-tensor-generated.h>")
    print("#include <torch-gobject/torch-util.h>")
    print("")
    print("template <typename T> using ArrayRef = c10::ArrayRef<T>;")
    print("using IntArrayRef = c10::IntArrayRef;")
    print("using Device = c10::Device;")
    print("using Dimname = at::Dimname;")
    print("using Generator = at::Generator;")
    print("using Layout = c10::Layout;")
    print("using MemoryFormat = c10::MemoryFormat;")
    print("using Scalar = c10::Scalar;")
    print("using ScalarType = c10::ScalarType;")
    print("using Storage = c10::Storage;")
    print("using Tensor = torch::Tensor;")
    print("using TensorList = at::TensorList;")
    print("using TensorOptions = c10::TensorOptions;")

    for d in declarations:
        print_function_body(d)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml", help="Path to Declarations.yaml")
    parser.add_argument("--header", action="store_true", help="Generating header")
    parser.add_argument("--output", help="File to write to")
    args = parser.parse_args()

    if args.output:
        sys.stdout = open(args.output, "wt")

    with open(args.yaml) as f:
        declarations = yaml.load(f, Loader=Loader)

    if args.header:
        print_header(declarations)
    else:
        print_source(declarations)


if __name__ == "__main__":
    main()

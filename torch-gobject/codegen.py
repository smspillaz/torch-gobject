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
        "convert_native_func": lambda a: "torch_device_get_real_device ({a})".format(a=a),
        "convert_gobject_prefix": lambda a: "g_autoptr ({a})".format(a=a.strip("* ")),
        "convert_gobject_func": lambda a: "torch_device_new_from_real_device ({a})".format(a=a),
    },
    "at::MemoryFormat": {
        "name": "TorchMemoryFormat",
        "convert_native_qualifiers": "",
        "convert_native_func": lambda a: "torch_memory_format_get_real_memory_format ({a})".format(a=a),
        "convert_gobject_prefix": lambda a: a,
        "convert_gobject_func": lambda a: "torch_memory_format_from_real_memory_format ({a})".format(a=a),
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
    "Storage": {
        "name": "TorchStorage *",
        "convert_native_qualifiers": "&",
        "convert_native_func": lambda a: "torch_storage_get_real_storage ({a})".format(a=a),
        "convert_gobject_prefix": lambda a: "g_autoptr ({a})".format(a=a.strip("* ")),
        "convert_gobject_func": lambda a: "torch_storage_new_from_real_storage ({a})".format(a=a),
    },
    "at::Tensor": {
        "name": "TorchTensor *",
        "convert_native_qualifiers": "&",
        "convert_native_func": lambda a: "torch_tensor_get_real_tensor ({a})".format(a=a),
        "convert_gobject_prefix": lambda a: "g_autoptr ({a})".format(a=a.strip("* ")),
        "convert_gobject_func": lambda a: "torch_tensor_new_from_real_tensor ({a})".format(a=a),
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
        "convert_native_func": lambda a: "torch_tensor_options_get_real_tensor_options ({a})".format(a=a),
        "convert_gobject_prefix": lambda a: "g_autoptr ({a})".format(a=a.strip("* ")),
        "convert_gobject_func": lambda a: "torch_tensor_options_new_from_real_tensor_options ({a})".format(a=a),
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
    }
}

RENAME_LIST = {
    "set_data": "set_data_from_tensor"
}

def non_namespaced_function_name(decl):
    inplace = ""
    overload = ""
    overload_name = ""

    if decl["name"].endswith("_"):
        inplace = "inplace"

    # Different from overload_name!
    if "overload" in decl:
        overload = decl["overload"].lower()

    if "overload_name" in decl:
        overload_name = decl["overload_name"].lower()

    return "_".join([x for x in [
        RENAME_LIST.get(decl["name"], decl["name"]).rstrip("_"),
        overload_name,
        overload,
        inplace
    ] if x])


def function_name(decl):
    method = ""

    if "Tensor" in decl["method_of"]:
        method = "tensor"

    return "_".join([x for x in [
        "torch",
        method,
        non_namespaced_function_name(decl)
    ] if x])


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
        return lambda a: "torch_array_ref_from_fixed_array({a}, {s})".format(a=a, s=type_spec["size"])

    return TYPE_MAPPING[unqualified]["convert_native_func"]

def map_element_type(type_spec):
    unqualified = unqualified_dynamic_type(type_spec["dynamic_type"])

    if unqualified == "IntArrayRef" and type_spec.get("size", None):
        return None

    return TYPE_MAPPING[unqualified].get("meta", {}).get("type", None)


def type_spec_to_gobject_type(type_spec):
    return {
        "name": type_spec["name"],
        "type": map_type_name(type_spec),
        "element-type": map_element_type(type_spec),
        "size": type_spec.get("size", None),
        "nullable": type_spec.get("is_nullable", False),
        "transfer": type_spec["transfer"]
    }


def determine_return_transfer_mode(decl):
    if (decl["schema_order_arguments"] and decl["schema_order_arguments"][0]["annotation"] == "a!"):
        return "self"
    elif "*" not in map_type_name(decl["returns"][0]):
        return "none"

    return "full"


def make_gobject_decl(decl):
    is_method = "namespace" not in decl["method_of"]
    out_return_parameter = None
    return_gobject = type_spec_to_gobject_type(dict(
        **decl["returns"][0],
        transfer=determine_return_transfer_mode(decl),
    )) if decl["returns"] else {
        "name": "",
        # If the function doesn't return anything,
        # it can still fail, so we need to return gboolean
        # here
        "type": "gboolean",
        "transfer": "none",
        "element-type": None,
        "size": None,
        "nullable": False
    }

    gobject_arguments = [
        type_spec_to_gobject_type(dict(**a, transfer="none"))
        for a in decl["arguments"]
    ]

    # We return a non-pointer type. Since calls into
    # C++ methods can throw exceptions, we need to convert
    # the return value into an outparam and make
    # the return value a gboolean indicating success
    # or failure
    if decl["returns"] and "*" not in return_gobject["type"]:
        return_gobject["name"] = return_gobject["name"] or "out_rv"
        out_return_parameter = return_gobject["name"]

        # The return value is now a pointer that gets passed in and set
        return_gobject["type"] = "{} *".format(return_gobject["type"])

        out_argument = dict(
            **return_gobject,
            out=True
        )
        out_argument["transfer"] = ""
        out_argument["element-type"] = ""

        gobject_arguments.append(out_argument)
        return_gobject = {
            "name": "",
            "type": "gboolean",
            "size": None,
            "transfer": "none",
            "element-type": None,
            "nullable": False
        }

    # Append an error argument to the parameters
    gobject_arguments.append({
        "type": "GError **",
        "name": "error",
        "nullable": True,
        "transfer": "full",
        "element-type": None,
        "out": True,
        "size": None
    })

    return {
        "name": function_name(decl),
        "is_method": is_method,
        "out_return_parameter": out_return_parameter,
        "returns": return_gobject,
        "arguments": gobject_arguments
    }


def make_gobject_decl_fwd_decl(decl):
    arg_str_list = []

    for a in decl["arguments"]:
        arg_str_list.append(a["type"] + " " + a["name"])

    return "".join([
        decl["returns"]["type"] + " ",
        decl["name"],
        " (",
        ", ".join(arg_str_list),
        ")"
    ])


def fmt_out(a):
    return "(out)" if a.get("out", False) else ""


def fmt_transfer(a):
    return "(transfer {a})".format(a="none" if a["transfer"] == "self" else a["transfer"]) if a["transfer"] and a["type"].endswith("*") else ""


def fmt_element_type(a):
    return "(element-type {a})".format(a=a["element-type"].strip(" *")) if a["element-type"] else ""


def fmt_array_fixed_size(a):
    return "(array fixed-size={a[size]})".format(a=a) if a["size"] else ""


def fmt_nullable(a):
    return "(nullable)".format(a=a) if a["nullable"] and "*" in a["type"] else ""


def fmt_annotations(a):
    annotations_str = " ".join([x for x in [
        fmt_out(a),
        fmt_transfer(a),
        fmt_element_type(a),
        fmt_array_fixed_size(a),
        fmt_nullable(a)
    ] if x])

    return ": {}".format(annotations_str) if annotations_str else ""


def make_gobject_decl_header(decl):
    return "\n".join([
        "/**",
        " * " + decl["name"] + ":",
    ] + [" * @{a[name]}{annotations}: A #{a[type]}".format(
        a=a,
        annotations=fmt_annotations(a)
    ) for a in decl["arguments"]] + [" *"] + (
        [" * Returns{annotations}: A #{ret[type]}".format(
            ret=decl["returns"],
            annotations=fmt_annotations(decl["returns"])
        )]
        if decl["returns"]["type"] != "void" else []
    ) + [" */"])


FUNCTION_BLACKLIST = (
    "data",
    "polygamma",  # declaration in Declarations.yaml is broken
)


def is_skipped(decl):
    if decl["name"] in FUNCTION_BLACKLIST:
        print("Skipped {decl[name]} - in blacklist".format(decl=decl), file=sys.stderr)
        return True

    if decl["name"] == "count_nonzero" and decl["overload_name"] == "":
        print("Skipped count_nonzero - is ambiguous", file=sys.stderr)
        return True

    # Skip internal funcs
    if decl["name"].startswith("_"):
        print("Skipped", decl["name"], " - is internal", file=sys.stderr)
        return True

    # Skip "out" versions
    if decl["name"].endswith("_out"):
        print("Skipped", decl["name"], " - is outfunc", file=sys.stderr)
        return True

    if decl["returns"]:
        if len(decl["returns"]) > 1:
            print("Skipped", decl["name"], "- is tuple-return", file=sys.stderr)
            return True

        if unqualified_dynamic_type(decl["returns"][0]["dynamic_type"]) not in TYPE_MAPPING:
            print("Skipped", decl["name"], "- returns", decl["returns"][0]["dynamic_type"], file=sys.stderr)
            return True

    for a in decl["arguments"]:
        if unqualified_dynamic_type(a["dynamic_type"]) not in TYPE_MAPPING:
            print("Skipped", decl["name"], "- takes", a["dynamic_type"], file=sys.stderr)
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
    wrapped_type = "c10::optional<{}>".format(unqualified_arg_type) if argument.get("is_nullable", False) else unqualified_arg_type
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


def make_function_call(decl, gobject_decl):
    out_return_parameter = [
        a for a in
        gobject_decl["arguments"]
        if a["name"] == gobject_decl["out_return_parameter"]
    ][0] if gobject_decl["out_return_parameter"] else None
    before_block = ""

    if "namespace" in decl["method_of"]:
        call = "torch::{name} ({args});".format(
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

    if decl["returns"]:
        return_type = decl["returns"][0]["dynamic_type"]
        unqualified_return_type = unqualified_dynamic_type(return_type)
        gobject_return_type = out_return_parameter["type"].strip(" *") if out_return_parameter else gobject_decl["returns"]["type"]

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

        if gobject_decl["returns"]["transfer"] == "self":
            convert_statement = ""
            return_statement = "return {};".format(gobject_decl["arguments"][0]["name"])
        else:
            call = " ".join([
                decl["returns"][0]["type"],
                "real_rv",
                "=",
                call
            ])

            convert_statement = " ".join([
                TYPE_MAPPING[unqualified_return_type]["convert_gobject_prefix"](gobject_return_type),
                "gobject_rv",
                "=",
                TYPE_MAPPING[unqualified_return_type]["convert_gobject_func"]("real_rv") + ";"
            ])

            # Cannot be "self", was checked earlier
            if gobject_decl["returns"]["transfer"] == "none":
                return_statement_operand = "gobject_rv;";
            elif gobject_decl["returns"]["transfer"] == "full":
                return_statement_operand = "static_cast <{}> (g_steal_pointer (&gobject_rv));".format(gobject_return_type)
            else:
                assert "Section not reachable" and False

            if out_return_parameter is not None:
                return_statement = "\n".join([
                    "*{} = {}".format(out_return_parameter["name"],
                                      return_statement_operand),
                    "return TRUE;"
                ])
            else:
                return_statement = "return {}".format(return_statement_operand)
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
            "g_set_error (error, G_IO_ERROR, G_IO_ERROR_FAILED, \"%s\", e.what ());",
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
    print("#include <torch-gobject/torch-errors.h>")
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

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
    "ArrayRef<double>": {
        "name": "GArray *",
        "meta": {
            "type": "double"
        },
        "convert_native_qualifiers": "",
        "convert_native_func": lambda a: "torch_array_ref_from_garray <double> ({a})".format(a=a),
        "convert_gobject_prefix": lambda a: "g_autoptr ({a})".format(a=a.strip("* ")),
        "convert_gobject_func": lambda a: "torch_new_garray_from_array_ref <double> ({a})".format(a=a),
    },
    "IntArrayRef": {
        "name": "GArray *",
        "meta": {
            "type": "long",
        },
        "convert_native_qualifiers": "",
        "convert_native_func": lambda a: "torch_array_ref_from_garray <long> ({a})".format(a=a),
        "convert_gobject_prefix": lambda a: "g_autoptr ({a})".format(a=a.strip("* ")),
        "convert_gobject_func": lambda a: "torch_new_garray_from_array_ref <long> ({a})".format(a=a),
    },
    "Device": {
        "name": "TorchDevice *",
        "convert_native_qualifiers": "&",
        "convert_native_func": lambda a: "torch_device_get_real_device ({a})".format(a=a),
        "convert_gobject_prefix": lambda a: "g_autoptr ({a})".format(a=a.strip("* ")),
        "convert_gobject_func": lambda a: "torch_device_new_from_real_device ({a})".format(a=a),
    },
    "MemoryFormat": {
        "name": "TorchMemoryFormat",
        "convert_native_qualifiers": "",
        "convert_native_func": lambda a: "torch_memory_format_get_real_memory_format ({a})".format(a=a),
        "convert_gobject_prefix": lambda a: a,
        "convert_gobject_func": lambda a: "torch_memory_format_from_real_memory_format ({a})".format(a=a),
    },
    "Scalar": {
        "name": "GValue *",
        "convert_native_qualifiers": "",
        "convert_native_func": lambda a: "torch_scalar_from_gvalue ({a})".format(a=a),
        "convert_gobject_prefix": lambda a: "g_autofree {a}".format(a=a),
        "convert_gobject_func": lambda a: "torch_gvalue_from_scalar ({a})".format(a=a),
    },
    "ScalarType": {
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
    "Tensor": {
        "name": "TorchTensor *",
        "convert_native_qualifiers": "&",
        "convert_native_func": lambda a: "torch_tensor_get_real_tensor ({a})".format(a=a),
        "convert_gobject_prefix": lambda a: "g_autoptr ({a})".format(a=a.strip("* ")),
        "convert_gobject_func": lambda a: "torch_tensor_new_from_real_tensor ({a})".format(a=a),
    },
    "TensorList": {
        "name": "GPtrArray *",
        "meta": {
            "type": "TorchTensor *"
        },
        "convert_native_qualifiers": "",
        "convert_native_func": lambda a: "torch_tensor_list_from_tensor_ptr_array ({a})".format(a=a),
        "convert_gobject_prefix": lambda a: "g_autoptr ({a})".format(a=a.strip("* ")),
        "convert_gobject_func": lambda a: "torch_tensor_ptr_array_from_tensor_list ({a})".format(a=a),
    },
    "TensorOptions": {
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


def map_type_name(type_spec):
    if type_spec["dynamic_type"] == "IntArrayRef" and type_spec.get("size", None):
        return "const long *"

    return TYPE_MAPPING[type_spec["dynamic_type"]]["name"]


def map_type_native_conv(type_spec):
    if type_spec["dynamic_type"] == "IntArrayRef" and type_spec.get("size", None):
        return lambda a: "torch_array_ref_from_fixed_array({a}, {s})".format(a=a, s=type_spec["size"])

    return TYPE_MAPPING[type_spec["dynamic_type"]]["convert_native_func"]

def map_element_type(type_spec):
    if type_spec["dynamic_type"] == "IntArrayRef" and type_spec.get("size", None):
        return None

    return TYPE_MAPPING[type_spec["dynamic_type"]].get("meta", {}).get("type", None)


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
    return {
        "name": function_name(decl),
        "is_method": "namespace" not in decl["method_of"],
        "returns": type_spec_to_gobject_type(dict(
            **decl["returns"][0],
            transfer=determine_return_transfer_mode(decl),
        )) if decl["returns"] else {
            "name": "",
            "type": "void",
            "size": None,
            "nullable": False
        },
        "arguments": [
            type_spec_to_gobject_type(dict(**a, transfer="none"))
            for a in decl["arguments"]
        ]
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


def is_skipped(decl):
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

        if decl["returns"][0]["dynamic_type"] not in TYPE_MAPPING:
            print("Skipped", decl["name"], "- returns", decl["returns"][0]["dynamic_type"], file=sys.stderr)
            return True

    for a in decl["arguments"]:
        if a["dynamic_type"] not in TYPE_MAPPING:
            print("Skipped", decl["name"], "- takes", a["dynamic_type"], file=sys.stderr)
            return True


def should_rewrite_for_scalar(decl):
    for a in decl["arguments"]:
        if a["dynamic_type"] == "Scalar":
            return True

    return False


def rewrite_decl_for_scalar(decl, scalar_type):
    d = deepcopy(decl)
    for a in d["arguments"]:
        if a["dynamic_type"] == "Scalar":
            a["dynamic_type"] = scalar_type

    d["name"] = non_namespaced_function_name(d)
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

    str_list.append(make_gobject_decl_header(gobject_decl))
    str_list.append(make_gobject_decl_fwd_decl(gobject_decl) + ";")

    print("\n")
    print("\n".join(str_list))


def make_argument_marshaller(argument, gobject_argument):
    arg_type = argument["dynamic_type"]

    return " ".join([
        arg_type,
        TYPE_MAPPING[arg_type]["convert_native_qualifiers"],
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
    call = "at::{name}({args});".format(
        name=decl["name"],
        args=", ".join([
            "real_" + a["name"] for a in decl["arguments"]
        ])
    )

    if decl["returns"]:
        return_type = decl["returns"][0]["dynamic_type"]
        gobject_return_type = TYPE_MAPPING[decl["returns"][0]["dynamic_type"]]["name"]

        call = " ".join([
            decl["returns"][0]["type"],
            "real_rv",
            "=",
            call
        ])

        if gobject_decl["returns"]["transfer"] == "self":
            convert_statement = ""
            return_statement = "return {};".format(gobject_decl["arguments"][0]["name"])
        else:
            convert_statement = " ".join([
                TYPE_MAPPING[return_type]["convert_gobject_prefix"](gobject_decl["returns"]["type"]),
                "gobject_rv",
                "=",
                TYPE_MAPPING[return_type]["convert_gobject_func"]("real_rv") + ";"
            ])

            if gobject_decl["returns"]["transfer"] == "none":
                return_statement = "return gobject_rv";
            elif gobject_decl["returns"]["transfer"] == "full":
                return_statement = "return g_steal_pointer(&gobject_rv);"
    else:
        convert_statement = ""
        return_statement = "return;"

    return "\n".join([
        call,
        convert_statement,
        return_statement
    ])


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
    str_list.append(make_gobject_decl_fwd_decl(gobject_decl))
    str_list.append("{")
    str_list.append(indent(make_argument_marshallers(decl["arguments"], gobject_decl["arguments"]), 4))
    str_list.append("");
    str_list.append(indent(make_function_call(decl, gobject_decl), 4));
    str_list.append("}")

    print("")
    print("\n".join(str_list))

def print_header(declarations):
    print("#include \"torch-device.h\"")
    print("#include \"torch-storage.h\"")
    print("#include \"torch-scalar-type.h\"")
    print("#include \"torch-memory-format.h\"")
    print("#include \"torch-tensor.h\"")
    print("#include \"torch-tensor-options.h\"")
    print("#include \"torch-tensor-type.h\"")

    for d in declarations:
        print_function_decl(d)


def print_source(declarations):
    print("#include <torch/torch.h>")
    print("#include \"torch-tensor.h\"")
    print("#include \"torch-tensor-internal.h\"")
    print("#include \"torch-tensor-generated.h\"")
    print("")
    print("using namespace at = aten;")
    print("")
    for d in declarations:
        print_function_body(d)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml", help="Path to Declarations.yaml")
    parser.add_argument("--header", action="store_true", help="Generating header")
    args = parser.parse_args()
    
    with open(args.yaml) as f:
        declarations = yaml.load(f, Loader=Loader)
 
    if args.header:
        print_header(declarations)
    else:
        print_source(declarations)


if __name__ == "__main__":
    main()

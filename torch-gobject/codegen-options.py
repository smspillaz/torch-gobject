import argparse
import re
import os
import sys

from copy import copy


_MATCH_OPTIONS_STRUCT = re.compile(
    "struct.*\s(?P<name>[A-Za-z][A-Za-z0-9]*?Options)(?=\s).*"
)
_MATCH_ARG = re.compile(
    "\s*TORCH_ARG\((?P<type>[\w_\<\>\:]+),[\s]*(?P<name>[\w_]+)\)(:?\s+=\s+(?P<init>[\w_\.\\:\(\)]+))?"
)
_MATCH_USING = re.compile(
    r"\s*using (?P<name>[A-Za-z][A-Za-z0-9_]*?)\s+=\s+(?P<parent>[A-Za-z][A-Za-z0-9_\:]*).*;"
)
_RE_CAMEL_CASE1 = re.compile(r"(.)([A-Z][a-z]+)")
_RE_CAMEL_CASE2 = re.compile(r"([a-z0-9])([A-Z])")
_RE_TEMPLATE = r = re.compile(
    r"(?P<template>[A-Za-z][A-Za-z0-9\:]*)\<(?P<name>[A-Za-z0-9][A-Za-z0-9\_\:\,\s\<\>]*)\>"
)
_MATCH_VARIANT_DEF = re.compile(r".*(using|typedef).*c10::variant")
_MATCH_FUNC = re.compile(r"\s*(?P<name>[A-Za-z_][\w_]*)\((?P<args>.*?)\).*")
_MATCH_FUNC_ARG = re.compile(
    r"(?P<type>[A-Za-z_][\w_\<\>]*)\s*(?P<name>[A-Za-z_][\w]*)(?:\s*=\s*(?P<default>[\w\\-]+)|.*)"
)

TYPE_CONVERSION = {
    ".*": {
        "bool": {"name": "gboolean"},
        "double": {"name": "double"},
        "float": {"name": "float"},
        "int": {"name": "int"},
        "long": {"name": "long"},
        "int64_t": {"name": "int64_t"},
        "Tensor": {"name": "TorchTensor *", "transfer": "none"},
        "torch::Tensor": {"name": "TorchTensor *", "transfer": "none"},
        "torch::Dtype": {"name": "GType", "transfer": "none"},
        "std::string": {"name": "const char *"},
        "namedshape_t": {
            "name": "GArray *",
            "meta": {"element-type": "TorchNamedShape"},
            "transfer": "none",
        },
        "distance_function_t": {"name": "TorchDistanceFunction"},
        "conv_padding_mode_t": {"name": "TorchConvPaddingMode"},
        "conv_padding_t": {"name": "TorchConvPadding"},
        "rnn_options_base_mode_t": {"name": "TorchRNNOptionsBaseMode"},
        "activation_t": {"name": "TorchTransformerActivation"},
        "AnyModule": {"name": "TorchAnyModule *"},
        "ExpandingArrayDouble": {
            "name": "GArray *",
            "meta": {"element-type": "double", "n-elements": None},
        },
    },
    "Adaptive.*PoolOptions": {"output_size_t": {"name": "unsigned int"}},
    "Conv.*Options": {
        "padding_t": {"name": "Torch{}Padding", "replace_with_context": True},
        "padding_mode_t": {"name": "Torch{}PaddingMode", "replace_with_context": True},
    },
    "Embedding(Bag|Func).*Options": {
        "EmbeddingBagMode": {"name": "TorchEmbeddingBagMode"}
    },
    ".*LossFuncOptions": {
        "reduction_t": {"name": "Torch{}Reduction", "replace_with_context": True}
    },
    ".*LossOptions": {
        "reduction_t": {"name": "Torch{}Reduction", "replace_with_context": True}
    },
    "PadFuncOptions": {"mode_t": {"name": "TorchPadFuncOptionsMode"}},
    "RNN.*Options": {
        "nonlinearity_t": {"name": "Torch{}Nonlinearity", "replace_with_context": True}
    },
    "GridSampleFuncOptions": {
        "mode_t": {"name": "TorchGridSampleFuncMode"},
        "padding_mode_t": {"name": "TorchGridSampleFuncPaddingMode"},
    },
    "Transformer.*Options": {
        "activation_t": {"name": "TorchTransfomerOptionsActivation"}
    },
    "UpsampleOptions": {"mode_t": {"name": "TorchUpsampleOptionsMode"}},
    "InterpolateFuncOptions": {"mode_t": {"name": "TorchInterpolateOptionsMode"}},
    "Transformer.*Options": {
        "TransformerDecoderLayer": {"name": "TorchTransformerDecoderLayer"},
        "TransformerEncoderLayer": {"name": "TorchTransformerEncoderLayer"},
    },
}
TYPE_CONVERSION_LENGTH_SORTED_KEYS = list(
    reversed(sorted(TYPE_CONVERSION.keys(), key=lambda k: len(k)))
)


def indent(text, indent):
    pad = " " * indent
    return "\n".join([pad + line for line in text.splitlines()])


def convert_type_internal(native_type, context):
    for k in TYPE_CONVERSION_LENGTH_SORTED_KEYS:
        if (k == ".*" and not context) or (context and re.match(k, context)):
            obj = TYPE_CONVERSION[k].get(native_type, None)

            if obj is not None:
                obj = obj.copy()

                if obj.get("replace_with_context", False):
                    obj["name"] = obj["name"].format(context)

                return obj

    raise RuntimeError(
        "Don't know how to handle type {} in context {}".format(native_type, context)
    )


def parse_template(native_type, context):
    m = _RE_TEMPLATE.match(native_type)

    if m != None:
        template = m.group("template")
        typename = m.group("name")

        if template == "c10::optional":
            info = parse_template(typename, context)
            if not "*" in info["name"]:
                info["name"] = "{} *".format(info["name"])
            info["nullable"] = True
            return info

        if template == "std::vector":
            info = parse_template(typename, context)
            return {
                "name": "GArray *" if not "*" in info["name"] else "GPtrArray *",
                "meta": {
                    "element-type": info["name"],
                    "nullable": info.get("nullable", False),
                },
            }

        if template == "ExpandingArray":
            if "," in typename:
                n, typename = typename
            else:
                n = typename
                typename = "int64_t"
            return {
                "name": "GArray *",
                "meta": {
                    "element-type": typename,
                    "n-elements": int(n) if n.isdigit() else None,
                },
            }

        raise RuntimeError("Don't know how to handle template {}".format(template))

    return convert_type_internal(native_type, context)


def convert_type(native_type, context):
    return parse_template(native_type, context)


def camel_case_to_snake_case(camel_cased):
    camel_cased = _RE_CAMEL_CASE1.sub(r"\1_\2", camel_cased)
    return _RE_CAMEL_CASE2.sub(r"\1_\2", camel_cased).upper()


class MultilineFunctionDeclParser(object):
    def __init__(self):
        super().__init__()
        self.lines_data = []

    def process_line(self, line):
        self.lines_data.append(line)

        return ":" in line or ";" in line

    def emit(self):
        concat_lines = " ".join(
            " ".join(map(lambda x: x.strip(), self.lines_data)).split()
        )

        func_match = _MATCH_FUNC.match(concat_lines)
        assert func_match is not None

        name = func_match.group("name").strip()
        args = func_match.group("args").strip()

        # Hopefully whatever we get as the arguments here isn't too complicated...
        if args:
            args = list(map(lambda x: x.strip(), args.split(",")))
            args = [_MATCH_FUNC_ARG.match(a).groupdict() for a in args]
        else:
            args = []

        return {"name": name, "args": args}


class OptionsStructParser(object):
    def __init__(self, name, namespace):
        super().__init__()
        self.name = name
        self.args = []
        self.constructors = []
        self.subparser = None
        self.namespace = namespace

    def process_line(self, line):
        if self.subparser is not None:
            if self.subparser.process_line(line):
                self.constructors.append(self.subparser.emit())
                self.subparser = None
                return False
        else:
            m = _MATCH_ARG.match(line)

            if m != None:
                self.args.append(
                    {
                        "type": m.group("type"),
                        "name": m.group("name"),
                        "init": m.group("init"),
                    }
                )

            # Matched a constructor, it may be over multiple lines
            if re.match("\s*{}\(.*\).*".format(self.name), line):
                self.subparser = MultilineFunctionDeclParser()

                # Call again, this time with subparser set
                self.process_line(line)

            return line.startswith("};")

    def emit(self):
        return {
            "type": "options_info",
            "data": {
                "struct": self.name,
                "args": self.args,
                "parent": None,
                "namespace": self.namespace,
            },
        }


class VariantStructParser(object):
    def __init__(self, line, context, namespace):
        super().__init__()
        self.lines_data = []
        self.done = False
        self.context = context
        self.namespace = namespace

    def process_line(self, line):
        if not self.done:
            self.lines_data.append(line)

        self.done = ";" in self.lines_data[-1]

        if self.done:
            return True

        return False

    def emit(self):
        assert self.done == True

        # Parse the result, first concatenate together the lines, normalizing whitespace
        concat_lines = " ".join(
            " ".join(map(lambda x: x.strip(), self.lines_data)).split()
        )
        if concat_lines.startswith("using"):
            name = re.match(r"using (?P<name>[\w_]+)", concat_lines).group("name")
        elif concat_lines.startswith("typedef"):
            name = re.match(r".*\s+(?P<name>[\w_]+);", concat_lines).group("name")
        else:
            raise RuntimeError(
                "Expected variant definition to start with using or typedef"
            )

        internal_args = map(
            lambda x: x.strip().lstrip(),
            re.match(r".*c10::variant<(?P<args>.*)>.*;", concat_lines)
            .group("args")
            .split(","),
        )
        enumtype_args = [a for a in internal_args if a.startswith("enumtype")]

        return {
            "type": "enum",
            "data": {
                "name": name,
                "context": self.context,
                "args": [
                    {"enumtype": a, "type": a.lstrip("enumtype::k")}
                    for a in enumtype_args
                ],
                "namespace": self.namespace,
            },
        }


def parse_header(contents):
    current_stack = []
    namespace_stack = []

    for line in contents.splitlines():
        m_options_struct = _MATCH_OPTIONS_STRUCT.match(line)
        m_using = _MATCH_USING.match(line)
        m_variant_start = _MATCH_VARIANT_DEF.match(line)

        if line.strip().startswith("namespace"):
            namespace_stack.append(line.strip().split()[1])
        if "} // namespace" in line.strip():
            # this seems to be a pattern, don't know if we can rely on it
            namespace_stack.pop()

        if m_options_struct != None:
            if current_stack:
                for parser in reversed(current_stack):
                    yield parser.emit()

            current_stack = [
                OptionsStructParser(
                    m_options_struct.group("name"), copy(namespace_stack)
                )
            ]
        elif line.startswith("struct"):
            if current_stack:
                for parser in reversed(current_stack):
                    yield parser.emit()

            current_stack = []
        elif m_variant_start != None:
            # Assuming that current[-1] is either an OptionsStructParser or not defined.
            #
            # If not defined, then there is no context.
            current_stack.append(
                VariantStructParser(
                    line,
                    current_stack[-1].name if current_stack else None,
                    copy(namespace_stack),
                )
            )
        elif m_using != None:
            if m_using.group("name").endswith("Options"):
                yield {
                    "type": "options_info",
                    "data": {
                        "struct": m_using.group("name"),
                        "args": [],
                        "parent": m_using.group("parent"),
                        "namespace": copy(namespace_stack),
                    },
                }
            elif current_stack:
                yield {
                    "type": "typedef",
                    "data": {
                        "alias": m_using.group("name"),
                        "name": m_using.group("parent").split("::")[-1],
                        "context": current_stack[-1].name,
                        "namespace": copy(namespace_stack),
                    },
                }

        if current_stack:
            if current_stack[-1].process_line(line):
                # We're done parsing this section, pop
                yield current_stack[-1].emit()
                current_stack.pop()

    if current_stack != None:
        for parser in reversed(current_stack):
            yield parser.emit()


def refine_options_info(options_info):
    camel_name = "Torch{}".format(options_info["struct"])
    upper_name = camel_case_to_snake_case(camel_name)
    lower_name = upper_name.lower()
    parent_camel = (
        "Torch{}".format(options_info["parent"])
        if options_info["parent"]
        else "GObject"
    )
    parent_upper = camel_case_to_snake_case(parent_camel)
    parent_upper_components = parent_upper.split("_")
    parent_gtype = "_".join(
        [
            parent_upper_components[0],
            "TYPE",
        ]
        + parent_upper_components[1:]
    )

    return {
        **options_info,
        "gobject": {
            "upper": upper_name,
            "lower": lower_name,
            "name": camel_name,
            "parent": parent_camel,
            "parent_gtype": parent_gtype,
            "args": [
                {
                    "type": convert_type(arg["type"], options_info["struct"])["name"],
                    "name": arg["name"],
                    "init": arg["init"],
                }
                for arg in options_info["args"]
            ],
            "construct_args": [
                {
                    "type": convert_type(arg["type"], options_info["struct"])["name"],
                    "name": arg["name"],
                }
                for arg in options_info["args"]
            ],
        },
    }


def refine_enum(enum):
    enum_name_camel = convert_type(enum["name"], enum["context"])["name"]
    upper_name = camel_case_to_snake_case(enum_name_camel)

    return {
        **enum,
        "gobject": {
            "name": enum_name_camel,
            "args": [
                {"type": "_".join([upper_name, a["type"].upper()])}
                for a in enum["args"]
            ],
        },
    }


def refine_typedef(typedef):
    return {
        **typedef,
        "gobject": {
            "name": convert_type(typedef["name"], typedef["context"])["name"],
            "alias": convert_type(typedef["alias"], typedef["context"])["name"],
        },
    }


def refine_parsing_stream(parsed_structs):
    for item in parsed_structs:
        if item["type"] == "options_info":
            yield {"type": "options_info", "data": refine_options_info(item["data"])}
        elif item["type"] == "enum":
            yield {
                "type": "enum",
                "data": refine_enum(item["data"]),
            }
        elif item["type"] == "typedef":
            yield {"type": "typedef", "data": refine_typedef(item["data"])}


def global_refine_step(parsed_structs):
    # Global refine step - make a map of all
    # the options_info - if we have one as a parent
    # of the other, then it should also copy its construct
    # arguments
    all_options_infos = {
        item["data"]["gobject"]["name"]: item["data"]
        for item in parsed_structs
        if item["type"] == "options_info"
    }

    for value in all_options_infos.values():
        if value["gobject"]["parent"] != "GObject":
            value["gobject"]["construct_args"].extend(
                all_options_infos[value["gobject"]["parent"]]["gobject"][
                    "construct_args"
                ]
            )

    return parsed_structs


def print_options_struct_header(options_info):
    upper_camel_case = "_".join(options_info["gobject"]["upper"].split("_")[1:])
    lower_camel_case = options_info["gobject"]["lower"]
    camel = options_info["gobject"]["name"]

    print(
        "#define TORCH_TYPE_{upper} {lower}_get_type ()".format(
            upper=upper_camel_case, lower=lower_camel_case
        )
    )
    print(
        "G_DECLARE_FINAL_TYPE({name}, {lower}, TORCH, {upper}, {parent})".format(
            name=camel,
            lower=lower_camel_case,
            upper=upper_camel_case,
            parent=options_info["gobject"]["parent"],
        )
    )
    print("")
    for arg in options_info["gobject"]["args"]:
        arg_type = arg["type"]
        arg_name = arg["name"]
        arg_init = arg["init"]
        print(
            "gboolean {lower}_get_{name} ({struct} *options, {type} {star}out_{name}, GError **error);".format(
                type=arg_type,
                name=arg_name,
                lower=lower_camel_case,
                struct=camel,
                star="*" if "*" not in arg_name else "*",
            )
        )
        print("")
        print(
            "gboolean {lower}_set_{name} ({struct} *options, {type} {name}, GError **error);".format(
                type=arg_type, name=arg_name, lower=lower_camel_case, struct=camel
            )
        )
        print("")

    print(
        "{struct} * {lower}_new ({args}GError **error);".format(
            struct=camel,
            lower=lower_camel_case,
            args=", ".join(
                [
                    "{type} {name}".format(type=arg["type"], name=arg["name"])
                    for arg in options_info["gobject"]["construct_args"]
                ]
            )
            + ", "
            if options_info["gobject"]["construct_args"]
            else "",
        )
    )
    print("")
    print(
        "{struct} * {lower}_new_floating ({args}GError **error);".format(
            struct=camel,
            lower=lower_camel_case,
            args=", ".join(
                [
                    "{type} {name}".format(type=arg["type"], name=arg["name"])
                    for arg in options_info["gobject"]["construct_args"]
                ]
            )
            + ", "
            if options_info["gobject"]["construct_args"]
            else "",
        )
    )
    print("")


def print_enum_header(enum):
    print(
        "typedef enum {{\n{args}\n}} {name};\n".format(
            name=enum["gobject"]["name"],
            args=indent(",\n".join([a["type"] for a in enum["gobject"]["args"]]), 2),
        )
    )


def print_typedef_header(typedef):
    print(
        "typedef {name} {alias};\n".format(
            name=typedef["gobject"]["name"], alias=typedef["gobject"]["alias"]
        )
    )


def print_header(parsed_structs):
    for parsed_struct in parsed_structs:
        if parsed_struct["type"] == "options_info":
            print_options_struct_header(parsed_struct["data"])
        if parsed_struct["type"] == "enum":
            print_enum_header(parsed_struct["data"])
        if parsed_struct["type"] == "typedef":
            print_typedef_header(parsed_struct["data"])


def parse_header_file_for_files(header_filename):
    header_dirname = os.path.dirname(header_filename)

    with open(header_filename, "r") as f:
        lines = f.readlines()

    matches = [re.match(r"#include <torch/nn/(?P<header>.*)>", line) for line in lines]

    return [
        os.path.join(header_dirname, m.group("header"))
        for m in matches
        if m is not None
    ]


def generate_header(headers_options):
    print("#include <glib-object.h>")
    print("#include <torch-gobject/torch-tensor.h>")
    print("#include <torch-gobject/torch-tensor-options.h>")
    print("")
    print("typedef int (*TorchDistanceFunction)(TorchTensor *src, TorchTensor *dst);")
    print("typedef struct _TorchTransformerDecoderLayer TorchTransformerDecoderLayer;")
    print("typedef struct _TorchTransformerEncoderLayer TorchTransformerEncoderLayer;")
    print("typedef struct _TorchAnyModule TorchAnyModule;")
    print("\nG_BEGIN_DECLS\n")

    for header_filename, options in headers_options:
        print("/* source: {} */".format(os.path.basename(header_filename)))
        print_header(options)

    print("\nG_END_DECLS")


def print_options_struct_source(options_info):
    upper_camel_case = "_".join(options_info["gobject"]["upper"].split("_")[1:])
    lower_camel_case = options_info["gobject"]["lower"]
    camel = options_info["gobject"]["name"]
    parent_gtype = options_info["gobject"]["parent_gtype"]

    print(
        "\n".join(
            [
                "struct _{}".format(camel),
                "{",
                indent(
                    "{type} parent;".format(type=options_info["gobject"]["parent"]), 2
                ),
                "};",
            ]
        )
    )
    print("")

    if "detail" not in options_info["namespace"]:
        print(
            "\n".join(
                [
                    "typedef struct _{}Private".format(camel),
                    "{",
                    indent(
                        "\n".join(
                            [
                                "{namespace}{type} *internal;".format(
                                    # This is fine to do since leading "::" means "global namespace"
                                    namespace="::".join(options_info["namespace"])
                                    + "::",
                                    type=options_info["struct"],
                                ),
                            ]
                        ),
                        2,
                    ),
                    "",
                    indent(
                        "\n".join(
                            [
                                "{type} {name};".format(type=a["type"], name=a["name"])
                                for a in options_info["gobject"]["args"]
                            ]
                        ),
                        2,
                    ),
                    "}} {}Private;".format(camel),
                ]
            )
        )
    else:
        # If this is a "detail" namespace we shouldn't actually
        # try to generate this object - just use it as a parent
        # for another object type and keep the properties there
        print(
            "\n".join(
                [
                    "typedef struct _{camel}Private {{}} {camel}Private;".format(
                        camel=camel
                    )
                ]
            )
        )

    print("")
    print("static void initable_iface_init (GInitableIface *iface);")
    print("")
    print(
        "G_DEFINE_TYPE_WITH_CODE ({camel}, {lower}, {parent_gtype}, G_ADD_PRIVATE ({camel}) G_IMPLEMENT_INTERFACE (G_TYPE_INITABLE, initable_iface_init))".format(
            camel=camel, lower=lower_camel_case, parent_gtype=parent_gtype
        )
    )
    print(
        "#define {upper}_GET_PRIVATE(a) static_cast <{camel}Private *> ({lower}_get_instance_private ((a)));".format(
            camel=camel,
            lower=lower_camel_case,
            upper=upper_camel_case,
        )
    )
    print("")


def print_enum_source(enum):
    print("")


def print_typedef_source(typedef):
    print("")


def print_source(parsed_structs):
    for parsed_struct in parsed_structs:
        if parsed_struct["type"] == "options_info":
            print_options_struct_source(parsed_struct["data"])
        if parsed_struct["type"] == "enum":
            print_enum_source(parsed_struct["data"])
        if parsed_struct["type"] == "typedef":
            print_typedef_source(parsed_struct["data"])


def generate_source(headers_options):
    print("#include <gio/gio.h>")
    print("#include <torch-gobject/torch-tensor.h>")
    print("#include <torch-gobject/torch-nn-options-generated.h>")
    print("")
    print("#include <string>")
    print("#include <vector>")
    print("#include <torch/nn/options.h>")
    print('#include "torch-enums.h"')

    for header_filename, options in headers_options:
        print("/* source: {} */".format(os.path.basename(header_filename)))
        print_source(options)


def generate_options_for_header_filename(header_filename):
    with open(header_filename) as f:
        options = list(parse_header(f.read()))
        options = list(refine_parsing_stream(options))
        options = global_refine_step(options)

    return options


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "options_header", help="The header file to parse for option headers"
    )
    parser.add_argument("--output", help="Where to write the file")
    parser.add_argument("--header", action="store_true", help="Writing a header file")
    parser.add_argument("--source", action="store_true", help="Writing a source file")
    args = parser.parse_args()

    if args.output:
        sys.stdout = open(args.output, "wt")

    headers_options = [
        (header_to_parse, generate_options_for_header_filename(header_to_parse))
        for header_to_parse in parse_header_file_for_files(args.options_header)
    ]

    if args.header:
        generate_header(headers_options)

    if args.source:
        generate_source(headers_options)


if __name__ == "__main__":
    main()

import argparse
import re
import os


_MATCH_OPTIONS_STRUCT = re.compile("struct TORCH_API (?P<name>[A-Za-z]+Options)(?=\s).*")
_MATCH_ARG = re.compile("\s*TORCH_ARG\((?P<type>[\w_]+),[\s]*(?P<name>[\w_]+)\)(:?\s+=\s+(?P<init>[\w_\.\\:\(\)]+))?")
_RE_CAMEL_CASE1 = re.compile(r"(.)([A-Z][a-z]+)")
_RE_CAMEL_CASE2 = re.compile(r"([a-z0-9])([A-Z])")


TYPE_CONVERSION = {
}


def camel_case_to_snake_case(camel_cased):
    camel_cased = _RE_CAMEL_CASE1.sub(r"\1_\2", camel_cased)
    return _RE_CAMEL_CASE2.sub(r"\1_\2", camel_cased).upper()

def parse_header(contents):
    current_struct = None
    current_args = []

    for line in contents.splitlines():
        m = _MATCH_OPTIONS_STRUCT.match(line)

        if m != None:
            if current_struct != None:
                yield {
                    "struct": current_struct,
                    "args": current_args
                }

            current_args = []
            current_struct = m.group("name")
        elif "struct" in line and current_struct != None:
            yield {
                "struct": current_struct,
                "args": current_args
            }
            current_struct = None


        m = _MATCH_ARG.match(line)

        if m != None and current_struct != None:
            current_args.append((
                m.group("type"), m.group("name"), m.group("init")
            ))


    yield {
        "struct": current_struct,
        "args": current_args
    }


def print_header(options):
    print("""
#include <glib-object.h>

G_BEGIN_DECLS
""")

    for options_info in options:
        upper_camel_case = camel_case_to_snake_case(options_info["struct"])
        lower_camel_case = upper_camel_case.lower()
        print("#define TORCH_TYPE_{upper} torch_{lower}_get_type ()".format(
            upper=upper_camel_case,
            lower=lower_camel_case
        ))
        print("G_DECLARE_FINAL_TYPE(Torch{name}, torch_{lower}, TORCH, {upper}, GObject)".format(
            name=options_info["struct"],
            lower=lower_camel_case,
            upper=upper_camel_case
        ))
        print("")

    print("G_END_DECLS")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="The header file to parse for options")
    parser.add_argument("--output", help="Where to write the file")
    parser.add_argument("--header", action="store_true", help="Writing a header file")
    args = parser.parse_args()

    with open(args.input) as f:
        options = list(parse_header(f.read()))

    if args.header:
        print_header(options)
    else
        print_source(options)

if __name__ == "__main__":
    main()

from clang.cindex import Index, CursorKind

FLOATING_TYPES = {
    'double': '__RP_1__',
    'float': '__RP_2__'
}

class CodeAnalyzer:
    def __init__(self):
        self.var_types = {}  
        self.var_locations = {} 

    def analyze(self, cursor, filename):
        """Collect variable declarations and their types."""
        if cursor.location.file and cursor.location.file.name != filename:
            return
        if cursor.kind == CursorKind.VAR_DECL:
            var_name = cursor.spelling
            if var_name:
                full_type = cursor.type.spelling
                self.var_types[var_name] = full_type
                self.var_locations[var_name] = (
                    filename,
                    cursor.location.line,
                    cursor.location.column
                )
        for child in cursor.get_children():
            self.analyze(child, filename)

def split_template_params(param_str):
    """Split template parameters, respecting nested templates."""
    params = []
    depth = 0
    current = ""
    for char in param_str:
        if char == '<':
            depth += 1
            current += char
        elif char == '>':
            depth -= 1
            current += char
        elif char == ',' and depth == 0:
            params.append(current.strip())
            current = ""
        else:
            current += char

    if current:
        params.append(current.strip())
    return params

def replace_floating_types(type_str):
    """Replace floating-point types within type strings, including templates."""
    if '<' not in type_str: 
        return FLOATING_TYPES.get(type_str, type_str)
    
    base_end = type_str.find('<')
    base = type_str[:base_end]
    params_str = type_str[base_end + 1:type_str.rfind('>')]
    params = split_template_params(params_str)
    new_params = [FLOATING_TYPES.get(p, p) for p in params]
    return f"{base}<{', '.join(new_params)}>"

def print_annotated_code(input_file, output_file=None):
    """Parse and print annotated code with floating-point type replacements."""
    index = Index.create()
    tu = index.parse(input_file, args=['-std=c++17'])
    if not tu:
        print("Failed to parse file")
        return

    analyzer = CodeAnalyzer()
    analyzer.analyze(tu.cursor, input_file)

    with open(input_file, 'r') as f:
        lines = f.readlines()

    newfile = list()
    print("\n Annotated Code:")
    for line_idx, line in enumerate(lines):
        modified_line = line.rstrip()
        for var in analyzer.var_locations:
            file, line_num, _ = analyzer.var_locations[var]
            orig_type = analyzer.var_types[var]
            new_type = replace_floating_types(orig_type)
 
            if orig_type in modified_line:
                modified_line = modified_line.replace(orig_type, new_type)

        print(modified_line)
        newfile.append(modified_line)

    if output_file is not None:
        with open(output_file, 'w') as f:
            for line in newfile:
                f.write(line)
                f.write("\n")

# with open("sample.cpp", "r") as f:
#     code = f.read()
#     

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        print_annotated_code(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 2:
        print_annotated_code(sys.argv[1])
    else:
        print("Please enter valid inputs")
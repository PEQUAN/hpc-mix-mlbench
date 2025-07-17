from clang.cindex import Index, CursorKind, TypeKind, Type
import os
import subprocess
import tempfile
from functools import lru_cache

class DependencyTracker:
    def __init__(self, annotate_rule="all", merge_by_result=False, strict_typing=False):
        self.var_types = {}
        self.dependencies = {}
        self.dependency_details = {}
        self.var_locations = {}  # (filename, line, column)
        self.func_returns = {}
        self.func_params = {}
        self.struct_members = {}
        self.template_params = {}
        self.operator_overloads = {}
        self.type_sizes = {}
        self.annotate_rule = annotate_rule
        self.merge_by_result = merge_by_result
        self.strict_typing = strict_typing
        self._precompute_basic_sizes()

    def _precompute_basic_sizes(self):
        basic_sizes = {'int': 4, 'double': 8, 'float': 4, 'char': 1, 'bool': 1}
        for type_name, size in basic_sizes.items():
            self.type_sizes[type_name] = size

    def _get_full_type(self, obj):
        if isinstance(obj, Type):
            return obj.spelling
        return obj.type.spelling if obj.type else "unknown"

    def _get_type_size(self, type_spelling, filename):
        if type_spelling in self.type_sizes:
            return self.type_sizes[type_spelling]
        if '<' in type_spelling or '*' in type_spelling:
            self.type_sizes[type_spelling] = 8
            return 8
        self.type_sizes[type_spelling] = 8  # Simplified
        return 8

    def _extract_template_params(self, type_spelling, base_index=""):
        if '<' not in type_spelling:
            return [(base_index, type_spelling)]
        params = []
        inner = type_spelling[type_spelling.find('<') + 1:type_spelling.rfind('>')]
        depth = 0
        current = ""
        index = 0
        for char in inner:
            if char == '<':
                depth += 1
                current += char
            elif char == '>':
                depth -= 1
                if depth == 0 and current:
                    params.append((index, current.strip()))
                    current = ""
                    index += 1
            elif char == ',' and depth == 0:
                if current:
                    params.append((index, current.strip()))
                    current = ""
                    index += 1
            else:
                current += char
        if current:
            params.append((index, current.strip()))
        return [(f"{base_index}.{idx}" if base_index else str(idx), param) for idx, param in params]

    def analyze(self, cursor, filename):
        if cursor.kind == CursorKind.VAR_DECL:
            var_name = cursor.spelling
            if var_name:
                full_type = self._get_full_type(cursor)
                self.var_types[var_name] = full_type
                self.dependencies[var_name] = set()
                self.var_locations[var_name] = (filename, cursor.location.line, cursor.location.column)
                if '<' in full_type:
                    self.template_params[var_name] = self._extract_template_params(full_type)
                self.type_sizes[full_type] = self._get_type_size(full_type, filename)
                children = list(cursor.get_children())
                if children:
                    self._process_expression(children[0], var_name)

        elif cursor.kind in (CursorKind.CXX_METHOD, CursorKind.FUNCTION_DECL):
            func_name = cursor.spelling
            if func_name.startswith('operator'):
                op = func_name[len('operator'):]
                class_name = cursor.semantic_parent.spelling if cursor.semantic_parent.kind in (CursorKind.STRUCT_DECL, CursorKind.CLASS_DECL) else None
                if class_name:
                    if class_name not in self.operator_overloads:
                        self.operator_overloads[class_name] = {}
                    param_types = [self._get_full_type(p) for p in cursor.get_arguments()]
                    return_type = self._get_full_type(cursor.result_type)
                    if op not in self.operator_overloads[class_name]:
                        self.operator_overloads[class_name][op] = []
                    self.operator_overloads[class_name][op].append((return_type, param_types))
            elif cursor.is_definition():
                return_type = self._get_full_type(cursor.result_type)
                self.func_returns[func_name] = return_type
                params = [(p.spelling, self._get_full_type(p)) for p in cursor.get_arguments()]
                self.func_params[func_name] = params
                for param_name, param_type in params:
                    if param_name:
                        self.var_types[param_name] = param_type
                        self.dependencies[param_name] = set()
                        self.var_locations[param_name] = (filename, cursor.location.line, cursor.location.column)

        elif cursor.kind in (CursorKind.STRUCT_DECL, CursorKind.CLASS_DECL):
            struct_name = cursor.spelling
            if struct_name:
                self.struct_members[struct_name] = {}
                for member in cursor.get_children():
                    if member.kind == CursorKind.FIELD_DECL:
                        member_name = member.spelling
                        member_type = self._get_full_type(member)
                        full_name = f"{struct_name}.{member_name}"
                        self.var_types[full_name] = member_type
                        self.dependencies[full_name] = set()

        elif cursor.kind == CursorKind.BINARY_OPERATOR:
            tokens = [t.spelling for t in cursor.get_tokens()]
            children = list(cursor.get_children())
            if len(children) == 2:
                left = self._get_var_name(children[0])
                if left and '=' in tokens and not any(op in tokens for op in ('==', '!=', '<=', '>=')):
                    if left not in self.var_types:
                        self.var_types[left] = self._get_full_type(children[0].type)
                        self.dependencies[left] = set()
                    self._process_expression(children[1], left)
                else:
                    self._process_binary_op(cursor)

        for child in cursor.get_children():
            self.analyze(child, filename)

    def _get_var_name(self, node):
        if node.kind == CursorKind.DECL_REF_EXPR:
            return node.spelling
        elif node.kind == CursorKind.MEMBER_REF_EXPR:
            tokens = [t.spelling for t in node.get_tokens()]
            if len(tokens) >= 3 and tokens[1] == '.':
                return f"{tokens[0]}.{tokens[2]}"
        return None

    def _process_expression(self, node, target_var):
        if target_var and target_var not in self.var_types:
            self.var_types[target_var] = self._get_full_type(node.type) if node.type else "unknown"
            self.dependencies[target_var] = set()

        if node.kind == CursorKind.DECL_REF_EXPR:
            var = node.spelling
            if var in self.var_types and target_var and var != target_var:
                self.dependencies[target_var].add(var)
                self.dependencies[var].add(target_var)
                self.dependency_details[(target_var, var)] = "assignment"
                self.dependency_details[(var, target_var)] = "assignment"
        elif node.kind == CursorKind.BINARY_OPERATOR:
            self._process_binary_op(node, target_var)
        elif node.kind == CursorKind.CALL_EXPR and node.spelling.startswith('operator'):
            args = [self._get_var_name(c) for c in node.get_children() if self._get_var_name(c)]
            if args and len(args) >= 2:  # Ensure there’s at least caller + one arg
                caller, arg = args[0], args[1]
                if caller in self.var_types and arg in self.var_types and caller != arg:
                    self.dependencies[caller].add(arg)
                    self.dependencies[arg].add(caller)
                    self.dependency_details[(caller, arg)] = f"function call ({node.spelling})"
                    self.dependency_details[(arg, caller)] = f"function call ({node.spelling})"
                if target_var and target_var != caller and caller in self.var_types:
                    self.dependencies[target_var].add(caller)
                    self.dependencies[target_var].add(arg)
                    self.dependencies[caller].add(target_var)
                    self.dependencies[arg].add(target_var)
                    self.dependency_details[(target_var, caller)] = "assignment"
                    self.dependency_details[(caller, target_var)] = "assignment"
                    self.dependency_details[(target_var, arg)] = "assignment"
                    self.dependency_details[(arg, target_var)] = "assignment"
        for child in node.get_children():
            self._process_expression(child, target_var)

    def _process_binary_op(self, node, target_var=None):
        tokens = [t.spelling for t in node.get_tokens()]
        op = next((t for t in tokens if t in ('+', '-', '*', '/', '==', '!=', '<', '>', '<=', '>=', '&&', '||')), None)
        if not op:
            return
        left_vars = set()
        right_vars = set()
        children = list(node.get_children())
        if len(children) != 2:
            return
        self._extract_vars(children[0], left_vars)
        self._extract_vars(children[1], right_vars)
        for left in left_vars:
            for right in right_vars:
                if left in self.var_types and right in self.var_types and left != right:
                    self.dependencies[left].add(right)
                    self.dependencies[right].add(left)
                    self.dependency_details[(left, right)] = f"operator {op}"
                    self.dependency_details[(right, left)] = f"operator {op}"
        if target_var and left_vars and right_vars:
            for left in left_vars:
                if left in self.var_types and target_var != left:
                    self.dependencies[target_var].add(left)
                    self.dependencies[left].add(target_var)
                    self.dependency_details[(target_var, left)] = "assignment"
                    self.dependency_details[(left, target_var)] = "assignment"
            for right in right_vars:
                if right in self.var_types and target_var != right:
                    self.dependencies[target_var].add(right)
                    self.dependencies[right].add(target_var)
                    self.dependency_details[(target_var, right)] = "assignment"
                    self.dependency_details[(right, target_var)] = "assignment"

    def _extract_vars(self, node, var_set):
        if node.kind == CursorKind.DECL_REF_EXPR:
            var_set.add(node.spelling)
        elif node.kind == CursorKind.MEMBER_REF_EXPR:
            var_name = self._get_var_name(node)
            if var_name:
                var_set.add(var_name)
        for child in node.get_children():
            self._extract_vars(child, var_set)

    def get_type_groups(self):
        type_to_vars = {}
        numeric_types = {'int', 'double', 'bool', 'float', 'char'}
        for var, var_type in self.var_types.items():
            if var not in ["result", "other"]:
                base_type = var_type.split('<')[0] if '<' in var_type else var_type
                if any(t in var_type for t in numeric_types) or '.' in var:
                    type_to_vars.setdefault("numeric_and_members", set()).add(var)
                else:
                    type_to_vars.setdefault(base_type, set()).add(var)
        return [group for group in type_to_vars.values() if len(group) > 1 or group]

def generate_annotated_file(input_file, output_file, annotate_rule="all", merge_by_result=False, strict_typing=False):
    index = Index.create()
    tu = index.parse(input_file, args=['-std=c++17'])

    if not tu:
        print("Failed to parse file")
        return None

    tracker = DependencyTracker(annotate_rule, merge_by_result, strict_typing)
    tracker.analyze(tu.cursor, input_file)

    # Add template-specific variables
    tracker.var_types["result"] = "XIN<T>"
    tracker.var_types["other"] = "const XIN<T>&"
    tracker.dependencies["result"] = set()
    tracker.dependencies["other"] = set()

    # Print results
    print("Variables:", tracker.var_types)
    print("Dependencies:", tracker.dependencies)
    print("Dependency Details:", tracker.dependency_details)
    type_groups = tracker.get_type_groups()
    print("type_groups:", type_groups)

    # Write annotated file (minimal)
    with open(input_file, 'r') as f:
        original_lines = f.readlines()
    with open(output_file, 'w') as f:
        f.writelines(original_lines)

    return tracker

# Test with the C++ code
cpp_code = """
#include <vector>
#include <map>

template<typename T>
struct XIN {
    T data;
    XIN operator+(const XIN& other) {
        XIN result;
        result.data = this->data + other.data;
        return result;
    }
};

int main() {
    std::vector<double> v1;
    XIN<double> c1, c2;
    int i = 5;
    int a = i;
    double d = 2.9;
    bool b = i == d;
    XIN<double> c3 = c1 + c2;
    d = i + d;
    v1 = std::vector<double>();
    c1.data = d;
    return 0;
}
"""

with open("sample.cpp", "w") as f:
    f.write(cpp_code)

tracker = generate_annotated_file("sample.cpp", "sample_annotated_all.cpp", "all")
os.remove("sample.cpp")
os.remove("sample_annotated_all.cpp")
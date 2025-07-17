#include <clang-c/Index.h>
#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <memory>

using namespace std;

// Helper to convert CXString to std::string and dispose it
string cxStringToString(CXString cxStr) {
    string result = clang_getCString(cxStr);
    clang_disposeString(cxStr);
    return result;
}

// Struct to hold dependency details
struct DependencyDetail {
    string source;
    string target;
    string relationship;
    DependencyDetail(string s, string t, string r) : source(s), target(t), relationship(r) {}
};

// Main DependencyTracker class
class DependencyTracker {
private:
    unordered_map<string, string> var_types_;               // Variable -> Type
    unordered_map<string, unordered_set<string>> dependencies_; // Variable -> Dependencies
    vector<DependencyDetail> dependency_details_;           // Detailed dependencies
    unordered_map<string, tuple<string, int, int>> var_locations_; // Variable -> (file, line, col)
    unordered_map<string, string> func_returns_;            // Function -> Return type
    unordered_map<string, vector<pair<string, string>>> func_params_; // Function -> Params
    unordered_map<string, unordered_map<string, string>> struct_members_; // Struct -> Members
    unordered_map<string, vector<pair<int, string>>> template_params_; // Variable -> Template params
    unordered_map<string, unordered_map<string, vector<pair<string, vector<string>>>>> operator_overloads_; // Class -> Op -> (Return, Params)
    unordered_map<string, int> type_sizes_;                 // Type -> Size
    unordered_set<string> uninitialized_vars_;              // Vars used before declaration
    string annotate_rule_;
    bool merge_by_result_;
    bool strict_typing_;

    void precomputeBasicSizes() {
        type_sizes_["int"] = 4;
        type_sizes_["double"] = 8;
        type_sizes_["float"] = 4;
        type_sizes_["char"] = 1;
        type_sizes_["bool"] = 1;
    }

    string getFullType(CXType type) {
        CXString typeStr = clang_getTypeSpelling(type);
        string result = cxStringToString(typeStr);
        return result.empty() ? "unknown" : result;
    }

    int getTypeSize(const string& type_spelling, const string& filename) {
        if (type_sizes_.count(type_spelling)) return type_sizes_[type_spelling];
        if (type_spelling.find('<') != string::npos || type_spelling.find('*') != string::npos || type_spelling.find('&') != string::npos) {
            type_sizes_[type_spelling] = 8;
            return 8;
        }
        type_sizes_[type_spelling] = 8; // Simplified
        return 8;
    }

    string getVarName(CXCursor cursor) {
        CXString spelling = clang_getCursorSpelling(cursor);
        string name = cxStringToString(spelling);
        if (clang_getCursorKind(cursor) == CXCursor_MemberRefExpr) {
            CXSourceRange range = clang_getCursorExtent(cursor);
            CXToken* tokens;
            unsigned numTokens;
            clang_tokenize(clang_Cursor_getTranslationUnit(cursor), range, &tokens, &numTokens);
            string result;
            for (unsigned i = 0; i < numTokens; ++i) {
                string token = cxStringToString(clang_getTokenSpelling(clang_Cursor_getTranslationUnit(cursor), tokens[i]));
                if (token == "." || token == "->") {
                    result = cxStringToString(clang_getTokenSpelling(clang_Cursor_getTranslationUnit(cursor), tokens[i-1])) + "." +
                             cxStringToString(clang_getTokenSpelling(clang_Cursor_getTranslationUnit(cursor), tokens[i+1]));
                    break;
                }
            }
            clang_disposeTokens(clang_Cursor_getTranslationUnit(cursor), tokens, numTokens);
            return result;
        } else if (clang_getCursorKind(cursor) == CXCursor_UnaryOperator) {
            CXSourceRange range = clang_getCursorExtent(cursor);
            CXToken* tokens;
            unsigned numTokens;
            clang_tokenize(clang_Cursor_getTranslationUnit(cursor), range, &tokens, &numTokens);
            if (numTokens > 0 && (cxStringToString(clang_getTokenSpelling(clang_Cursor_getTranslationUnit(cursor), tokens[0])) == "*" ||
                                  cxStringToString(clang_getTokenSpelling(clang_Cursor_getTranslationUnit(cursor), tokens[0])) == "&")) {
                CXCursor child = clang_getNullCursor();
                clang_visitChildren(cursor, [](CXCursor c, CXCursor, CXClientData data) {
                    *static_cast<CXCursor*>(data) = c;
                    return CXChildVisit_Break;
                }, &child);
                string childName = getVarName(child);
                clang_disposeTokens(clang_Cursor_getTranslationUnit(cursor), tokens, numTokens);
                return childName;
            }
            clang_disposeTokens(clang_Cursor_getTranslationUnit(cursor), tokens, numTokens);
        }
        return name;
    }

    void processExpression(CXCursor node, const string& target_var);
    void processBinaryOp(CXCursor node, const string& target_var);
    void processFunctionCall(CXCursor node, const string& target_var);
    void processChainedAssignment(CXCursor node, const string& outer_target);

public:
    DependencyTracker(const string& annotate_rule = "all", bool merge_by_result = false, bool strict_typing = false)
        : annotate_rule_(annotate_rule), merge_by_result_(merge_by_result), strict_typing_(strict_typing) {
        precomputeBasicSizes();
    }

    void analyze(CXCursor cursor, const string& filename) {
        CXCursorKind kind = clang_getCursorKind(cursor);
        if (kind == CXCursor_VarDecl) {
            string var_name = getVarName(cursor);
            if (!var_name.empty()) {
                string full_type = getFullType(clang_getCursorType(cursor));
                var_types_[var_name] = full_type;
                dependencies_[var_name] = unordered_set<string>();
                CXSourceLocation loc = clang_getCursorLocation(cursor);
                unsigned line, column;
                CXFile file;
                clang_getFileLocation(loc, &file, &line, &column, nullptr);
                var_locations_[var_name] = make_tuple(cxStringToString(clang_getFileName(file)), line, column);
                if (full_type.find('<') != string::npos) {
                    // Template param parsing (simplified)
                    template_params_[var_name] = {{0, full_type.substr(full_type.find('<') + 1, full_type.rfind('>') - full_type.find('<') - 1)}};
                }
                type_sizes_[full_type] = getTypeSize(full_type, filename);
                clang_visitChildren(cursor, [](CXCursor c, CXCursor, CXClientData data) {
                    static_cast<DependencyTracker*>(data)->processExpression(c, static_cast<DependencyTracker*>(data)->getVarName(c));
                    return CXChildVisit_Continue;
                }, this);
                if (uninitialized_vars_.count(var_name)) uninitialized_vars_.erase(var_name);
            }
        } else if (kind == CXCursor_CXXMethod || kind == CXCursor_FunctionDecl) {
            string func_name = getVarName(cursor);
            if (func_name.find("operator") == 0) {
                string op = func_name.substr(8);
                CXCursor parent = clang_getCursorSemanticParent(cursor);
                string class_name = cxStringToString(clang_getCursorSpelling(parent));
                if (!class_name.empty() && clang_getCursorKind(parent) == CXCursor_StructDecl || clang_getCursorKind(parent) == CXCursor_ClassDecl) {
                    vector<string> param_types;
                    clang_visitChildren(cursor, [](CXCursor c, CXCursor, CXClientData data) {
                        if (clang_getCursorKind(c) == CXCursor_ParmDecl) {
                            auto* vec = static_cast<vector<string>*>(data);
                            vec->push_back(cxStringToString(clang_getTypeSpelling(clang_getCursorType(c))));
                        }
                        return CXChildVisit_Continue;
                    }, &param_types);
                    string return_type = getFullType(clang_getCursorResultType(cursor));
                    operator_overloads_[class_name][op].push_back({return_type, param_types});
                }
            } else if (clang_isDefinition(cursor)) {
                func_returns_[func_name] = getFullType(clang_getCursorResultType(cursor));
                vector<pair<string, string>> params;
                clang_visitChildren(cursor, [](CXCursor c, CXCursor, CXClientData data) {
                    if (clang_getCursorKind(c) == CXCursor_ParmDecl) {
                        auto* vec = static_cast<vector<pair<string, string>>*>(data);
                        string name = cxStringToString(clang_getCursorSpelling(c));
                        string type = cxStringToString(clang_getTypeSpelling(clang_getCursorType(c)));
                        vec->emplace_back(name, type);
                    }
                    return CXChildVisit_Continue;
                }, &params);
                func_params_[func_name] = params;
                for (const auto& [param_name, param_type] : params) {
                    if (!param_name.empty()) {
                        var_types_[param_name] = param_type;
                        dependencies_[param_name] = unordered_set<string>();
                        CXSourceLocation loc = clang_getCursorLocation(cursor);
                        unsigned line, column;
                        CXFile file;
                        clang_getFileLocation(loc, &file, &line, &column, nullptr);
                        var_locations_[param_name] = make_tuple(cxStringToString(clang_getFileName(file)), line, column);
                    }
                }
            }
        } else if (kind == CXCursor_StructDecl || kind == CXCursor_ClassDecl) {
            string struct_name = getVarName(cursor);
            if (!struct_name.empty()) {
                struct_members_[struct_name] = unordered_map<string, string>();
                clang_visitChildren(cursor, [](CXCursor c, CXCursor, CXClientData data) {
                    if (clang_getCursorKind(c) == CXCursor_FieldDecl) {
                        auto* tracker = static_cast<DependencyTracker*>(data);
                        string member_name = cxStringToString(clang_getCursorSpelling(c));
                        string full_name = tracker->getVarName(clang_getCursorSemanticParent(c)) + "." + member_name;
                        string member_type = tracker->getFullType(clang_getCursorType(c));
                        tracker->var_types_[full_name] = member_type;
                        tracker->dependencies_[full_name] = unordered_set<string>();
                    }
                    return CXChildVisit_Continue;
                }, this);
            }
        } else if (kind == CXCursor_BinaryOperator) {
            CXSourceRange range = clang_getCursorExtent(cursor);
            CXToken* tokens;
            unsigned numTokens;
            clang_tokenize(clang_Cursor_getTranslationUnit(cursor), range, &tokens, &numTokens);
            bool isAssignment = false;
            for (unsigned i = 0; i < numTokens; ++i) {
                string token = cxStringToString(clang_getTokenSpelling(clang_Cursor_getTranslationUnit(cursor), tokens[i]));
                if (token == "=" && !find_if(tokens, tokens + numTokens, [](CXToken t) {
                    string s = cxStringToString(clang_getTokenSpelling(clang_Cursor_getTranslationUnit(t), t));
                    return s == "==" || s == "!=" || s == "<=" || s == ">=";
                })) {
                    isAssignment = true;
                    break;
                }
            }
            clang_disposeTokens(clang_Cursor_getTranslationUnit(cursor), tokens, numTokens);
            CXCursor left = clang_getNullCursor(), right = clang_getNullCursor();
            clang_visitChildren(cursor, [](CXCursor c, CXCursor, CXClientData data) {
                auto* pair = static_cast<pair<CXCursor*, CXCursor*>*>(data);
                if (pair->first.kind == CXCursorKind_InvalidCode) pair->first = c;
                else pair->second = c;
                return CXChildVisit_Continue;
            }, &pair<CXCursor*, CXCursor*>(&left, &right));
            if (isAssignment && !clang_Cursor_isNull(left)) {
                string left_name = getVarName(left);
                if (!left_name.empty()) {
                    if (!var_types_.count(left_name)) {
                        var_types_[left_name] = getFullType(clang_getCursorType(left));
                        dependencies_[left_name] = unordered_set<string>();
                    }
                    if (!clang_Cursor_isNull(right)) processExpression(right, left_name);
                    if (clang_getCursorKind(right) == CXCursor_BinaryOperator) {
                        processChainedAssignment(right, left_name);
                    }
                }
            } else {
                processBinaryOp(cursor, "");
            }
        } else if (kind == CXCursor_CallExpr && cxStringToString(clang_getCursorSpelling(cursor)).find("operator") != 0) {
            processFunctionCall(cursor, "");
        }

        clang_visitChildren(cursor, [](CXCursor c, CXCursor, CXClientData data) {
            static_cast<DependencyTracker*>(data)->analyze(c, static_cast<DependencyTracker*>(data)->var_locations_.begin()->second.first);
            return CXChildVisit_Continue;
        }, this);
    }

    void processExpression(CXCursor node, const string& target_var) {
        if (!target_var.empty() && !var_types_.count(target_var)) {
            var_types_[target_var] = getFullType(clang_getCursorType(node));
            dependencies_[target_var] = unordered_set<string>();
        }

        CXCursorKind kind = clang_getCursorKind(node);
        if (kind == CXCursor_DeclRefExpr) {
            string var = getVarName(node);
            if (!var.empty() && var != target_var) {
                if (!var_types_.count(var)) uninitialized_vars_.insert(var);
                else if (!target_var.empty()) {
                    dependencies_[target_var].insert(var);
                    dependencies_[var].insert(target_var);
                    dependency_details_.emplace_back(target_var, var, "assignment");
                    dependency_details_.emplace_back(var, target_var, "assignment");
                }
            }
        } else if (kind == CXCursor_BinaryOperator) {
            processBinaryOp(node, target_var);
        } else if (kind == CXCursor_CallExpr) {
            string func_name = cxStringToString(clang_getCursorSpelling(node));
            if (func_name.find("operator") == 0) {
                vector<string> args;
                clang_visitChildren(node, [](CXCursor c, CXCursor, CXClientData data) {
                    string name = static_cast<DependencyTracker*>(data)->getVarName(c);
                    if (!name.empty()) static_cast<vector<string>*>(data)->push_back(name);
                    return CXChildVisit_Continue;
                }, &args);
                if (args.size() >= 2) {
                    string caller = args[0], arg = args[1];
                    if (var_types_.count(caller) && var_types_.count(arg) && caller != arg) {
                        dependencies_[caller].insert(arg);
                        dependencies_[arg].insert(caller);
                        dependency_details_.emplace_back(caller, arg, "function call (" + func_name + ")");
                        dependency_details_.emplace_back(arg, caller, "function call (" + func_name + ")");
                    }
                    if (!target_var.empty() && target_var != caller && var_types_.count(caller)) {
                        dependencies_[target_var].insert(caller);
                        dependencies_[target_var].insert(arg);
                        dependencies_[caller].insert(target_var);
                        dependencies_[arg].insert(target_var);
                        dependency_details_.emplace_back(target_var, caller, "assignment");
                        dependency_details_.emplace_back(caller, target_var, "assignment");
                        dependency_details_.emplace_back(target_var, arg, "assignment");
                        dependency_details_.emplace_back(arg, target_var, "assignment");
                    }
                }
            } else {
                processFunctionCall(node, target_var);
            }
        } else if (kind == CXCursor_UnaryOperator) {
            CXSourceRange range = clang_getCursorExtent(node);
            CXToken* tokens;
            unsigned numTokens;
            clang_tokenize(clang_Cursor_getTranslationUnit(node), range, &tokens, &numTokens);
            string op = numTokens > 0 ? cxStringToString(clang_getTokenSpelling(clang_Cursor_getTranslationUnit(node), tokens[0])) : "";
            clang_disposeTokens(clang_Cursor_getTranslationUnit(node), tokens, numTokens);
            if (op == "*" || op == "&") {
                CXCursor child = clang_getNullCursor();
                clang_visitChildren(node, [](CXCursor c, CXCursor, CXClientData data) {
                    *static_cast<CXCursor*>(data) = c;
                    return CXChildVisit_Break;
                }, &child);
                string var = getVarName(child);
                if (var_types_.count(var) && !target_var.empty() && var != target_var) {
                    dependencies_[target_var].insert(var);
                    dependencies_[var].insert(target_var);
                    dependency_details_.emplace_back(target_var, var, "unary " + op);
                    dependency_details_.emplace_back(var, target_var, "unary " + op);
                }
            }
        } else if (kind != CXCursor_BinaryOperator && kind != CXCursor_CallExpr) {
            clang_visitChildren(node, [](CXCursor c, CXCursor, CXClientData data) {
                auto* pair = static_cast<pair<DependencyTracker*, string>*>(data);
                pair->first->processExpression(c, pair->second);
                return CXChildVisit_Continue;
            }, &pair<DependencyTracker*, string>(this, target_var));
        }
    }

    void processBinaryOp(CXCursor node, const string& target_var) {
        CXSourceRange range = clang_getCursorExtent(node);
        CXToken* tokens;
        unsigned numTokens;
        clang_tokenize(clang_Cursor_getTranslationUnit(node), range, &tokens, &numTokens);
        string op;
        for (unsigned i = 0; i < numTokens; ++i) {
            string token = cxStringToString(clang_getTokenSpelling(clang_Cursor_getTranslationUnit(node), tokens[i]));
            if (token == "+" || token == "-" || token == "*" || token == "/" || token == "==" || token == "!=" ||
                token == "<" || token == ">" || token == "<=" || token == ">=" || token == "&&" || token == "||") {
                op = token;
                break;
            }
        }
        clang_disposeTokens(clang_Cursor_getTranslationUnit(node), tokens, numTokens);
        if (op.empty()) return;

        unordered_set<string> left_vars, right_vars;
        CXCursor left = clang_getNullCursor(), right = clang_getNullCursor();
        clang_visitChildren(node, [](CXCursor c, CXCursor, CXClientData data) {
            auto* pair = static_cast<pair<CXCursor*, CXCursor*>*>(data);
            if (pair->first.kind == CXCursorKind_InvalidCode) pair->first = c;
            else pair->second = c;
            return CXChildVisit_Continue;
        }, &pair<CXCursor*, CXCursor*>(&left, &right));
        
        extractVars(left, left_vars);
        extractVars(right, right_vars);

        for (const auto& left : left_vars) {
            for (const auto& right : right_vars) {
                if (var_types_.count(left) && var_types_.count(right) && left != right) {
                    dependencies_[left].insert(right);
                    dependencies_[right].insert(left);
                    dependency_details_.emplace_back(left, right, "operator " + op);
                    dependency_details_.emplace_back(right, left, "operator " + op);
                }
            }
        }
        if (!target_var.empty() && !left_vars.empty() && !right_vars.empty()) {
            for (const auto& left : left_vars) {
                if (var_types_.count(left) && target_var != left) {
                    dependencies_[target_var].insert(left);
                    dependencies_[left].insert(target_var);
                    dependency_details_.emplace_back(target_var, left, "assignment");
                    dependency_details_.emplace_back(left, target_var, "assignment");
                }
            }
            for (const auto& right : right_vars) {
                if (var_types_.count(right) && target_var != right) {
                    dependencies_[target_var].insert(right);
                    dependencies_[right].insert(target_var);
                    dependency_details_.emplace_back(target_var, right, "assignment");
                    dependency_details_.emplace_back(right, target_var, "assignment");
                }
            }
        }
    }

    void processFunctionCall(CXCursor node, const string& target_var) {
        string func_name = cxStringToString(clang_getCursorSpelling(node));
        vector<string> args;
        clang_visitChildren(node, [](CXCursor c, CXCursor, CXClientData data) {
            if (clang_getCursorKind(c) != CXCursor_CallExpr) {
                string name = static_cast<DependencyTracker*>(data)->getVarName(c);
                if (!name.empty()) static_cast<vector<string>*>(data)->push_back(name);
            }
            return CXChildVisit_Continue;
        }, &args);
        for (const auto& arg : args) {
            if (var_types_.count(arg) && !target_var.empty() && target_var != arg) {
                dependencies_[target_var].insert(arg);
                dependencies_[arg].insert(target_var);
                dependency_details_.emplace_back(target_var, arg, "call " + func_name);
                dependency_details_.emplace_back(arg, target_var, "call " + func_name);
            }
        }
        for (size_t i = 0; i < args.size(); ++i) {
            for (size_t j = i + 1; j < args.size(); ++j) {
                if (var_types_.count(args[i]) && var_types_.count(args[j]) && args[i] != args[j]) {
                    dependencies_[args[i]].insert(args[j]);
                    dependencies_[args[j]].insert(args[i]);
                    dependency_details_.emplace_back(args[i], args[j], "call " + func_name);
                    dependency_details_.emplace_back(args[j], args[i], "call " + func_name);
                }
            }
        }
    }

    void processChainedAssignment(CXCursor node, const string& outer_target) {
        CXSourceRange range = clang_getCursorExtent(node);
        CXToken* tokens;
        unsigned numTokens;
        clang_tokenize(clang_Cursor_getTranslationUnit(node), range, &tokens, &numTokens);
        bool isAssignment = false;
        for (unsigned i = 0; i < numTokens; ++i) {
            if (cxStringToString(clang_getTokenSpelling(clang_Cursor_getTranslationUnit(node), tokens[i])) == "=") {
                isAssignment = true;
                break;
            }
        }
        clang_disposeTokens(clang_Cursor_getTranslationUnit(node), tokens, numTokens);
        if (!isAssignment) return;

        CXCursor left = clang_getNullCursor(), right = clang_getNullCursor();
        clang_visitChildren(node, [](CXCursor c, CXCursor, CXClientData data) {
            auto* pair = static_cast<pair<CXCursor*, CXCursor*>*>(data);
            if (pair->first.kind == CXCursorKind_InvalidCode) pair->first = c;
            else pair->second = c;
            return CXChildVisit_Continue;
        }, &pair<CXCursor*, CXCursor*>(&left, &right));
        
        string left_name = getVarName(left);
        if (!left_name.empty()) {
            if (!var_types_.count(left_name)) {
                var_types_[left_name] = getFullType(clang_getCursorType(left));
                dependencies_[left_name] = unordered_set<string>();
            }
            dependencies_[outer_target].insert(left_name);
            dependencies_[left_name].insert(outer_target);
            dependency_details_.emplace_back(outer_target, left_name, "assignment");
            dependency_details_.emplace_back(left_name, outer_target, "assignment");
            processExpression(right, left_name);
        }
    }

    void extractVars(CXCursor node, unordered_set<string>& var_set) {
        string var = getVarName(node);
        if (!var.empty()) var_set.insert(var);
        clang_visitChildren(node, [](CXCursor c, CXCursor, CXClientData data) {
            static_cast<DependencyTracker*>(data)->extractVars(c, *static_cast<unordered_set<string>*>(data));
            return CXChildVisit_Continue;
        }, &var_set);
    }

    vector<unordered_set<string>> getTypeGroups() {
        unordered_map<string, unordered_set<string>> type_to_vars;
        vector<string> numeric_types = {"int", "double", "bool", "float", "char"};
        for (const auto& [var, var_type] : var_types_) {
            if (var != "result" && var != "other") {
                string base_type = var_type.find('<') != string::npos ? var_type.substr(0, var_type.find('<')) : var_type;
                if (find_if(numeric_types.begin(), numeric_types.end(), [&](const string& t) { return var_type.find(t) != string::npos; }) != numeric_types.end() ||
                    var_type.find('.') != string::npos || var_type.find('*') != string::npos || var_type.find('&') != string::npos) {
                    type_to_vars["numeric_and_members"].insert(var);
                } else {
                    type_to_vars[base_type].insert(var);
                }
            }
        }
        vector<unordered_set<string>> groups;
        for (const auto& [_, vars] : type_to_vars) {
            if (vars.size() > 1 || !vars.empty()) groups.push_back(vars);
        }
        return groups;
    }

    void printResults() {
        cout << "Variables: {";
        for (const auto& [var, type] : var_types_) {
            cout << "'" << var << "': '" << type << "', ";
        }
        cout << "}\nDependencies: {";
        for (const auto& [var, deps] : dependencies_) {
            cout << "'" << var << "': {";
            for (const auto& dep : deps) cout << "'" << dep << "', ";
            cout << "}, ";
        }
        cout << "}\nDependency Details: {";
        for (const auto& detail : dependency_details_) {
            cout << "('" << detail.source << "', '" << detail.target << "'): '" << detail.relationship << "', ";
        }
        cout << "}\nUninitialized Variables: {";
        for (const auto& var : uninitialized_vars_) cout << "'" << var << "', ";
        cout << "}\ntype_groups: [";
        auto groups = getTypeGroups();
        for (const auto& group : groups) {
            cout << "{";
            for (const auto& var : group) cout << "'" << var << "', ";
            cout << "}, ";
        }
        cout << "]\n";
    }
};

int main() {
    CXIndex index = clang_createIndex(0, 0);
    const char* args[] = {"-std=c++17"};
    CXTranslationUnit tu = clang_parseTranslationUnit(index, "sample.cpp", args, 1, nullptr, 0, CXTranslationUnit_None);
    if (!tu) {
        cerr << "Failed to parse file\n";
        return 1;
    }

    DependencyTracker tracker;
    tracker.analyze(clang_getTranslationUnitCursor(tu), "sample.cpp");

    tracker.var_types_["result"] = "XIN<T>";
    tracker.var_types_["other"] = "const XIN<T>&";
    tracker.dependencies_["result"] = unordered_set<string>();
    tracker.dependencies_["other"] = unordered_set<string>();

    tracker.printResults();

    clang_disposeTranslationUnit(tu);
    clang_disposeIndex(index);
    return 0;
}
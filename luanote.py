from typing import Any
from lark import Lark, Token, Tree

with open("grammar.lark") as f:
    grammar = f.read()

parser = Lark(grammar)

class Type:
    def __init__(self, name: str):
        self.name = name
        self.alias = None
    def set_alias(self, alias: str):
        self.alias = alias
    def __repr__(self):
        if self.alias: return self.alias
        return self.name

class TableType(Type):
    def __init__(self,
            name: str,
            key: Type,
            value: Type,
            fields: dict[str | int, Type] | None = None):
        super().__init__(name)
        self.key = key
        self.value = value
        self.fields = fields
    def __repr__(self):
        if self.alias: return self.alias
        if self.key.name == "integer":
            return f"{self.value}[]"
        if self.fields and len(self.fields) > 0:
            return f"{{ {', '.join(f'{k}: {v}' for k, v in self.fields.items())} }}"
        return f"{{ [{self.key}]: {self.value} }}"

class FunctionType(Type):
    def __init__(self, name: str, params: list[Type], returns: Type):
        super().__init__(name)
        self.params = params
        self.returns = returns
    def __repr__(self):
        if self.alias: return self.alias
        return f"({', '.join(map(str, self.params))}) -> {self.returns}"

_LUAENV: dict[str, Type] = {
    "print": FunctionType("function", [Type("any")], Type("nil")),
    "tostring": FunctionType("function", [Type("any")], Type("string")),
}

def extends(type_a: Type, type_b: Type) -> bool:
    if type_b.name == "any":
        return True
    if type_a.name == "integer" and type_b.name == "number":
        return True
    if isinstance(type_a, TableType) and isinstance(type_b, TableType):
        if type_a.fields is not None and type_b.fields is not None:
            for key, value in type_a.fields.items():
                if key not in type_b.fields:
                    return False
                if not extends(value, type_b.fields[key]):
                    return False
            return True
        if not extends(type_a.key, type_b.key):
            return False
        if not extends(type_a.value, type_b.value):
            return False
    if isinstance(type_a, FunctionType) and isinstance(type_b, FunctionType):
        if len(type_a.params) != len(type_b.params):
            return False
        for a, b in zip(type_a.params, type_b.params):
            if not extends(b, a):
                return False
        if not extends(type_a.returns, type_b.returns):
            return False
        return True
    if type_a.name == type_b.name:
        return True
    return False

def loc(tree: Tree | Token) -> str:
    if isinstance(tree, Token):
        return f"test.lua:{tree.line}:{tree.column}:"
    if tree.meta.empty: return "???:?:?:"
    return f"test.lua:{tree.meta.start_pos}:{tree.meta.column}:"

type_alias_env: dict[str, Type] = {}

def infer(tree: Tree | Token, env: dict[str, Type] = _LUAENV) -> Type:
    global type_alias_env
    if isinstance(tree, Token):
        if tree.type == "NUMBER":
            return Type("number")
        if tree.type == "INTEGER":
            return Type("integer")
        if tree.type == "STRING":
            return Type("string")
        if tree.type == "BOOLEAN":
            return Type("boolean")
        if tree.type == "NIL":
            return Type("nil")
        if tree.type == "NAME":
            return env[tree.value]
        if tree.type == "PRIMITIVE_TYPE":
            return Type(tree.value)
        if tree.type == "TYPE_NAME":
            return type_alias_env[tree.value]
        assert False, f"Not implemented: {tree.type}"
    if tree.data == "unary_expr":
        op, value = tree.children
        value = infer(value, env)
        if op == "-":
            if value.name == "number":
                return Type("number")
            if value.name == "integer":
                return Type("integer")
        if op == "#":
            if value.name in ["string", "table"]:
                return Type("integer")
        if op == "not":
            if value.name == "boolean":
                return Type("boolean")
        assert False
    if tree.data in ["exp_expr", "mul_expr", "add_expr", "rel_expr", "eq_expr", "log_expr"]:
        left, op, right = tree.children
        op_loc = loc(op)
        left = infer(left, env)
        right = infer(right, env)
        if op in ["+", "-", "*", "/", "^"]:
            if left.name in ["number", "integer"] and right.name in ["number", "integer"]:
                if left.name == "number" or right.name == "number":
                    return Type("number")
                return Type("integer")
            raise ValueError(f"{op_loc} Invalid operation '{op}' between types '{left}' and '{right}'")
        if op == "..":
            if left.name == "string" and right.name == "string":
                return Type("string")
            raise ValueError(f"{op_loc} Invalid operation '{op}' between types '{left}' and '{right}'")
        if op in ["<", "<=", ">", ">="]:
            if left.name in ["number", "integer"] and right.name in ["number", "integer"]:
                return Type("boolean")
            raise ValueError(f"{op_loc} Invalid operation '{op}' between types '{left}' and '{right}'")
        if op in ["==", "~="]:
            if left.name == right.name:
                return Type("boolean")
        if op in ["and", "or"]:
            if left.name == "boolean" and right.name == "boolean":
                return Type("boolean")
            raise ValueError(f"{op_loc} Invalid operation '{op}' between types '{left}' and '{right}'")
        assert False, f"Not implemented: {op}"
    if tree.data == "table":
        fields = {}
        index = 1
        field_list = tree.children[0]
        new_array = []
        is_array = True
        different_keys = False
        different_values = False
        key_type = None
        value_type = None
        for field in field_list.children:
            if isinstance(field, Tree) and len(field.children) == 2:
                is_array = False
                key, value = field.children
                if isinstance(key, Tree):
                    cur_key_type = infer(key, env)
                else:
                    cur_key_type = Type("string")
                if key_type is None:
                    key_type = cur_key_type
                if not extends(cur_key_type, key_type) and not extends(key_type, cur_key_type):
                    different_keys = True
                value = infer(value, env)
                if value_type is None:
                    value_type = value
                if not extends(value, value_type) and not extends(value_type, value):
                    different_values = True
                fields[key] = value
                continue
            field = infer(field, env)
            fields[index] = field
            new_array.append(field)
            index = index + 1
        if is_array:
            if different_keys:
                raise ValueError(f"{loc(tree)} Table contains keys of different types: '{key_type}' and '{key}'")
            if different_values:
                raise ValueError(f"{loc(tree)} Table contains values of different types: '{value_type}' and '{value}'")
            first = new_array[0]
            for field in new_array:
                if not extends(field, first) and not extends(first, field):
                    raise ValueError(f"{loc(tree)} Array contains elements of different types: '{first}' and '{field}'")
            return TableType("table", Type("integer"), first, fields)
        assert key_type is not None and value_type is not None
        return TableType("table", key_type, value_type,  fields)
    if tree.data == "index_expr":
        table, index = tree.children
        index_loc = loc(index)
        table = infer(table, env)
        index = infer(index, env)
        if isinstance(table, TableType):
            if extends(index, table.key):
                return table.value
            raise ValueError(f"{index_loc} Invalid index type '{index}' for table with key type '{table.key}'")
        raise ValueError(f"{index_loc} Attempting to index a non-table type: '{table}'")
    if tree.data == "prop_expr":
        table, prop = tree.children
        prop_loc = loc(prop)
        table = infer(table, env)
        assert isinstance(prop, Token)
        prop = prop.value
        if isinstance(table, TableType):
            if table.fields is not None:
                if prop in table.fields:
                    return table.fields[prop]
                raise ValueError(f"{prop_loc} Table '{table}' does not contain property '{prop}'")
            if table.key.name == "string":
                return table.value
            raise ValueError(f"{prop_loc} Invalid property type '{prop}' for table with value type '{table.value}'")
        raise ValueError(f"{prop_loc} Attempting to access a property of a non-table type: '{table}'")
    if tree.data == "func_call":
        prefix, args = tree.children
        prefix_loc = loc(prefix)
        prefix = infer(prefix, env)
        if isinstance(args, Token):
            args = [infer(args, env)]
        elif args.data == "table":
            args = [infer(args, env)]
        else:
            args = [infer(arg, env) for arg in args.children]
        if isinstance(prefix, FunctionType):
            if len(args) != len(prefix.params):
                raise ValueError(f"{prefix_loc} Invalid number of arguments")
            for i, (arg, param) in enumerate(zip(args, prefix.params)):
                if not extends(arg, param):
                    raise ValueError(f"{prefix_loc} Invalid argument type, expected '{param}' but got '{arg}'")
            return prefix.returns
        raise ValueError(f"{prefix_loc} Attempting to call a non-function type: '{prefix}'")
    if tree.data == "func_body":
        if len(tree.children) == 1:
            params = Tree("param_list", [])
            body = tree.children[0]
        else:
            params, body = tree.children
        new_env = env.copy()
        for param in params.children:
            new_env[str(param)] = Type("any")
        return FunctionType("function", [Type("any") for _ in params.children], infer(body, new_env))
    if tree.data == "func_expr":
        return infer(tree.children[0], env)
    if tree.data == "var_decl":
        varlist, exprlist = tree.children
        for var, expr in zip(varlist.children, exprlist.children):
            env[str(var)] = infer(expr, env)
        return Type("nil")
    if tree.data == "func_decl":
        # TODO: add support for variadic functions
        type, name, body = tree.children
        new_env = env.copy()
        func_params, func_body = body.children
        func_params = func_params.children[0]
        for param in func_params.children:
            new_env[str(param)] = Type("any")
        for param in type.children:
            pname, ptype = param.children
            ptype = infer(ptype, env)
            new_env[str(pname)] = ptype
        str_name = ""
        if len(name.children) == 1:
            str_name = name.children[0]
            env[str(str_name)] = FunctionType("function", [
                new_env[str(param)] for param in func_params.children
            ], infer(func_body, new_env))
        return Type("nil")
    if tree.data == "chunk":
        new_env = env.copy()
        for statement in tree.children:
            type = infer(statement, new_env)
            if statement.data == "return_stmt":
                return type
        return Type("nil")        
    if tree.data == "return_stmt":
        # TODO: allow multiple return types
        return infer(tree.children[0].children[0], env)
    if tree.data == "array_type":
        return TableType("table", Type("integer"), infer(tree.children[0], env), {})
    if tree.data == "dict_type":
        return TableType("table", infer(tree.children[0], env), infer(tree.children[1], env), {})
    if tree.data == "obj_type":
        fields = {}
        for field in tree.children:
            name, type = field.children
            fields[str(name)] = infer(type, env)
        return TableType("table", Type("string"), Type("any"), fields)
    if tree.data == "alias_type":
        name, type = tree.children
        type = infer(type, env)
        type.set_alias(str(name))
        type_alias_env[str(name)] = type
        return type
    if tree.data == "record_type":
        name, *fields = tree.children
        fields: Any = {
            str(field.children[0]): infer(field.children[1], env) 
            for field in fields
        }
        type = TableType("table", Type("string"), Type("any"), fields)
        type_alias_env[str(name)] = type
        return type
    assert False, f"Not implemented: {tree.data}"

with open("test.lua") as f:
    code = f.read()

print(infer(parser.parse(code)))
from ctypes import Union
from typing import Callable, Optional
from lark import Lark, Token, Tree

with open("grammar.lark") as f:
    grammar = f.read()

parser = Lark(grammar)

type Node = Tree | Token

def get_loc(tree: Node):
    if isinstance(tree, Token):
        return f"test.lua:{tree.line}:{tree.column}:"
    if not tree.meta.empty:
        return f"test.lua:{tree.meta.line}:{tree.meta.column}:"
    if tree.children:
        return get_loc(tree.children[0])
    return "test.lua:0:0:"

type Type = PrimitiveType | PlaceHolderType | DictType | ObjType | UnionType | FuncType

class PrimitiveType:
    def __init__(self, name, value: Optional[str] = None):
        self.name = name
        self.value = value
    def __repr__(self):
        return self.name

class PlaceHolderType:
    def __init__(self, name):
        self.name = name
    def __repr__(self):
        return self.name
    
def eval_type(type_env: dict[str, Type], type: Type) -> Type:
    if isinstance(type, PlaceHolderType) and type.name in type_env:
        return type_env[type.name]
    return type
    
class DictType:
    def __init__(self, key: Type, value: Type):
        self.key = key
        self.value = value
    def __repr__(self):
        return f"{{[{self.key}]: {self.value}}}"

class ObjType:
    def __init__(self, fields: dict[str, Type]):
        self.fields = fields
    def __repr__(self):
        return f"{{{', '.join(f'{k}: {v}' for k, v in self.fields.items())}}}"
    
class UnionType:
    def __init__(self, values: list[Type]):
        self.values = values
    def simplify(self) -> 'UnionType':
        new_values = []
        for value in self.values:
            for new_value in new_values:
                if extends(value, new_value):
                    break
                if extends(new_value, value):
                    new_values.remove(new_value)
            else:
                new_values.append(value)
        return UnionType(new_values)
    def __repr__(self):
        return " | ".join(map(str, self.simplify().values))

class FuncType:
    def __init__(self, params: list[Type], ret: Type):
        self.params = params
        self.ret = ret
    def __repr__(self):
        return f"({', '.join(map(str, self.params))}) -> {self.ret}"

def extends(type_a: Type, type_b: Type, **kwargs) -> bool:
    if isinstance(type_a, PrimitiveType) and type_a.name == "never":
        return True
    if isinstance(type_b, PrimitiveType) and type_b.name == "unknown":
        return True
    if isinstance(type_a, PrimitiveType) and isinstance(type_b, PrimitiveType):
        if type_a.name == "integer" and type_b.name == "number":
            return True
        return type_a.name == type_b.name
    if isinstance(type_a, UnionType):
        return all(extends(v, type_b, **kwargs) for v in type_a.values)
    if isinstance(type_b, UnionType):
        return any(extends(type_a, v, **kwargs) for v in type_b.values)
    if isinstance(type_a, DictType) and isinstance(type_b, DictType):
        if not extends(type_a.key, type_b.key, **kwargs): return False
        if not extends(type_a.value, type_b.value, **kwargs): return False
        return True
    if isinstance(type_a, ObjType) and isinstance(type_b, ObjType):
        for k, v in type_a.fields.items():
            if k not in type_b.fields or not extends(v, type_b.fields[k], **kwargs):
                return False
        return True
    if isinstance(type_a, FuncType) and isinstance(type_b, FuncType):
        if len(type_a.params) != len(type_b.params): return False
        if not extends(type_a.ret, type_b.ret, **kwargs): return False
        return all(extends(a, b, **kwargs) for a, b in zip(type_a.params, type_b.params))
    return False

def typecheck_integer(value, **kwargs):
    return PrimitiveType("integer", value)

def typecheck_number(value, **kwargs):
    return PrimitiveType("number", value)

def typecheck_string(value, **kwargs):
    return PrimitiveType("string", value)

def typecheck_boolean(value, **kwargs):
    return PrimitiveType("boolean", value)

def typecheck_nil(value, **kwargs):
    return PrimitiveType("nil", value)

def typecheck_name(value, **kwargs):
    env = kwargs["env"]
    if str(value) in env:
        return env[value]
    raise TypeError(f"{get_loc(value)} Accessing unbound variable '{value}'")

def typecheck_type_name(value, **kwargs):
    type_env = kwargs["type_env"]
    if value in type_env:
        return type_env[value]
    return PlaceHolderType(value)

def typecheck_unary_expr(op: Token, value: Node, **kwargs):
    value_type = typecheck(value, **kwargs)
    assert isinstance(value_type, PrimitiveType) # TODO: add support for other types
    match op:
        case "-":
            if value_type.name == "integer":
                return PrimitiveType("integer")
            if value_type.name == "number":
                return PrimitiveType("number")
            raise TypeError(f"{get_loc(op)} Attempt to perform arithmetic (unary -) on a non-numeric value")
        case "#":
            # TODO: add support for length of tables
            if value_type.name == "string":
                return PrimitiveType("integer")
            raise TypeError(f"{get_loc(op)} Attempt to obtain length of a non-string value")
        case "not":
            if value_type.name == "boolean":
                return PrimitiveType("boolean")
            raise TypeError(f"{get_loc(op)} Attempt to perform logical (not) on a non-boolean value")
    assert False, f"Unknown unary operator {op}"

def typecheck_binary_expr(left: Node, op: Token, right: Node, **kwargs):
    left_type = typecheck(left, **kwargs)
    right_type = typecheck(right, **kwargs)
    assert isinstance(left_type, PrimitiveType) # TODO: add support for other types
    assert isinstance(right_type, PrimitiveType) # TODO: add support for other types
    match str(op):
        case x if x in "+-*/^%":
            if left_type.name == "integer" and right_type.name == "integer":
                return PrimitiveType("integer")
            if left_type.name == "number" and right_type.name == "number":
                return PrimitiveType("number")
            raise TypeError(f"{get_loc(op)} Attempt to perform arithmetic ({op}) on non-numeric values: '{left_type}' and '{right_type}'")
        case "..":
            if left_type.name == "string" and right_type.name == "string":
                return PrimitiveType("string")
            raise TypeError(f"{get_loc(op)} Attempt to concatenate non-string values: '{left_type}' and '{right_type}'")
        case x if x in ["==", "~=", "<", "<=", ">", ">="]:
            if left_type.name == right_type.name:
                return PrimitiveType("boolean")
            raise TypeError(f"{get_loc(op)} Attempt to compare values of different types: '{left_type}' and '{right_type}'")
        case x if x in ["and", "or"]:
            return UnionType([left_type, right_type])
    assert False, f"Unknown binary operator {op}"

def typecheck_array(fields: Tree, **kwargs):
    last_field = PrimitiveType("never")
    for field in fields.children:
        if len(field.children) == 2:
            _, value = field.children
        else: value = field.children[0]
        field_type = typecheck(value, **kwargs)
        if not extends(field_type, last_field, **kwargs):
            raise TypeError(f"{get_loc(field)} Attempt to create a table with non-homogeneous types")
    return DictType(PrimitiveType("integer"), last_field)

def typecheck_obj(fields: Tree, **kwargs):
    obj_fields = {}
    for field in fields.children:
        key, value = field.children
        value_type = typecheck(value, **kwargs)
        obj_fields[str(key)] = value_type
    return ObjType(obj_fields)

def typecheck_dict(fields: Tree, **kwargs):
    last_key_type = PrimitiveType("never")
    last_value_type = PrimitiveType("never")
    for field in fields.children:
        key, value = field.children
        key_type = typecheck(key, **kwargs)
        value_type = typecheck(value, **kwargs)
        if not extends(last_key_type, key_type, **kwargs):
            raise TypeError(f"{get_loc(key)} Attempt to create a table with different key types: '{key_type}' and '{last_key_type}'")
        if not extends(last_value_type, value_type, **kwargs):
            raise TypeError(f"{get_loc(value)} Attempt to create a table with non-homogeneous value types")
        last_key_type, last_value_type = key_type, value_type
    return DictType(key_type, value_type)

def typecheck_table(fields: Tree, **kwargs):
    is_array = True
    is_obj = True
    for field in fields.children:
        if len(field.children) == 2:
            key, value = field.children
            key_type = typecheck(key, **kwargs)
            value_type = typecheck(value, **kwargs)
            if not isinstance(key_type, PrimitiveType) or key_type.name != "integer":
                is_array = False
            if not isinstance(key_type, PrimitiveType) or key_type.name == "string":
                is_obj = True
    if is_array: return typecheck_array(fields, **kwargs)
    if is_obj: return typecheck_obj(fields, **kwargs)
    return typecheck_dict(fields, **kwargs)

def typecheck_type_alias(name, generics, type_expr, **kwargs):
    type_env = kwargs["type_env"]
    if name in type_env:
        raise TypeError(f"{get_loc(name)} Attempt to redefine type '{name}'")
    type_env[str(name)] = typecheck(type_expr, **kwargs)
    return type_env[str(name)]

def typecheck_primitive_type(name, **kwargs):
    return PrimitiveType(name)

def typecheck_obj_type(*fields, **kwargs):
    obj_fields = {}
    for field in fields:
        key, value = field.children
        value_type = typecheck(value, **kwargs)
        obj_fields[str(key)] = value_type
    return ObjType(obj_fields)

def typecheck_return_stmt(*values, **kwargs):
    if len(values) == 0:
        return PrimitiveType("nil")
    value = values[0]
    return typecheck(value, **kwargs)

def typecheck_func_call(name, args, **kwargs):
    name_loc = get_loc(name)
    name = typecheck(name, **kwargs)
    arg_locs = [get_loc(arg) for arg in args.children]
    args = [typecheck(arg, **kwargs) for arg in args.children]
    if not isinstance(name, FuncType):
        raise TypeError(f"{name_loc} Attempting to call a non-function value")
    if len(args) != len(name.params):
        raise TypeError(f"{name_loc} Attempting to call a function that requires {len(name.params)} parameters with {len(args)} arguments")
    i = 0
    for arg, param in zip(args, name.params):
        if not extends(arg, param, **kwargs):
            raise TypeError(f"{arg_locs[i]} Invalid argument type, expected '{param}', got '{arg}'")
        i += 1
    return name.ret

def typecheck_func_body(params, types, body, **kwargs):
    params = params.children[0]
    new_params = []
    for param, type in zip(params.children, types.children):
        pname, ptype = type.children
        assert param.type == "NAME" and param.value == pname.value
        p = typecheck(ptype, **kwargs)
        new_params.append(p)
        kwargs["env"][str(param)] = p
    return FuncType(new_params, typecheck(body, **kwargs))

def typecheck_func_decl(name, body, **kwargs):
    assert name.data == "func_name"
    name = name.children[0]
    body_type = typecheck(body, **kwargs)
    kwargs["env"][str(name)] = body_type
    return PrimitiveType("nil")

def typecheck_chunk(*stmts, **kwargs):
    new_env = kwargs["env"].copy()
    last = PrimitiveType("nil")
    kwargs["env"] = new_env
    for stmt in stmts:
        type = typecheck(stmt, **kwargs)
        if stmt.data == "return_stmt":
            last = type
    return last

DATA_TO_TYPECHECKER: dict[str, Callable[..., Type]] = {
    "INTEGER": typecheck_integer,
    "NUMBER": typecheck_number,
    "STRING": typecheck_string,
    "BOOLEAN": typecheck_boolean,
    "NIL": typecheck_nil,
    "NAME": typecheck_name,
    "TYPE_NAME": typecheck_type_name,
    "unary_expr": typecheck_unary_expr,
    "exp_expr": typecheck_binary_expr,
    "mul_expr": typecheck_binary_expr,
    "add_expr": typecheck_binary_expr,
    "rel_expr": typecheck_binary_expr,
    "eq_expr": typecheck_binary_expr,
    "log_expr": typecheck_binary_expr,
    "table": typecheck_table,
    "type_alias": typecheck_type_alias,
    "PRIMITIVE_TYPE": typecheck_primitive_type,
    "obj_type": typecheck_obj_type,
    "return_stmt": typecheck_return_stmt,
    "func_call": typecheck_func_call,
    "func_body": typecheck_func_body,
    "func_decl": typecheck_func_decl,
    "chunk": typecheck_chunk,
}

def typecheck(tree: Node, **kwargs):
    if isinstance(tree, Token):
        return DATA_TO_TYPECHECKER[tree.type](tree, **kwargs)
    return DATA_TO_TYPECHECKER[tree.data](*tree.children, **kwargs)

_LUAENV = {
    "print": FuncType([PrimitiveType("unknown")], PrimitiveType("nil")),
}

def main():
    type_env = {}
    env = _LUAENV
    with open("test.lua") as f:
        tree = parser.parse(f.read())
        program = typecheck(tree, env=env, type_env=type_env)
        print(program)

if __name__ == "__main__":
    main()
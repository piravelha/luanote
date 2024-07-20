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

type Type = PrimitiveType | PlaceHolderType | DictType | ObjType | UnionType | FuncType | PlaceHolderFuncType | GenericVar | GenericAlias | MetaType

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
    
class PlaceHolderFuncType:
    iota = 0
    def __init__(self, params: Tree, ret: Node):
        self.params = params
        self.ret = ret
        self.cur_iota = PlaceHolderFuncType.iota
        PlaceHolderFuncType.iota += 1
    def __repr__(self):
        return f"<fn:{self.cur_iota}>"
    
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
    
class MetaType:
    def __init__(self, base: Type, mt: Type):
        self.base = base
        self.mt = mt
    def __repr__(self):
        return f"meta<{self.base}, {self.mt}>"

class UnionType:
    def __init__(self, values: list[Type]):
        self.values = values
    def simplify(self, **kwargs) -> 'UnionType':
        new_values = []
        for value in self.values:
            for new_value in new_values:
                if not kwargs: continue
                if extends(value, new_value, **kwargs):
                    break
                if extends(new_value, value, **kwargs):
                    new_values.remove(new_value)
            else:
                new_values.append(value)
        if len(new_values) == 1:
            return new_values[0]
        return UnionType(new_values)
    def __repr__(self):
        return " | ".join(map(str, self.simplify().values))

class FuncType:
    def __init__(self, params: list[Type], ret: Type, generics: list[str] = []):
        self.params = params
        self.ret = ret
        self.generics = generics
    def __repr__(self):
        return f"({', '.join(map(str, self.params))}) -> {self.ret}"

class GenericVar:
    def __init__(self, name: str):
        self.name = name
    def __repr__(self):
        return self.name

class GenericAlias:
    def __init__(self, type: Type, generics: list[str]):
        self.type = type
        self.generics = generics
    def __repr__(self):
        return f"alias<{', '.join(self.generics)}>: {self.type}"

def type_error(loc: str, msg: str):
    print(f"\033[31mTYPE ERROR ({loc}):\033[0m")
    msg = "\033[31m-> \033[0m" + msg.replace("\n->", "\n\033[31m->\033[0m")
    print("" + msg)
    exit(1)

def simplify(type: Type, **kwargs) -> Type:
    if isinstance(type, UnionType):
        return type.simplify(**kwargs)
    return type

def apply(type: Type, subs: dict[str, Type], **kwargs) -> Type:
    if isinstance(type, GenericVar) and type.name in subs:
        return subs[type.name]
    if isinstance(type, PlaceHolderType) and type.name in subs:
        return subs[type.name]
    if isinstance(type, PlaceHolderFuncType) and repr(type) in subs:
        return subs[repr(type)]
    if isinstance(type, PrimitiveType):
        return type
    if isinstance(type, DictType):
        return DictType(apply(type.key, subs), apply(type.value, subs))
    if isinstance(type, ObjType):
        return ObjType({k: apply(v, subs) for k, v in type.fields.items()})
    if isinstance(type, MetaType):
        return MetaType(apply(type.base, subs), apply(type.mt, subs))
    if isinstance(type, UnionType):
        return UnionType([apply(v, subs) for v in type.values])
    if isinstance(type, FuncType):
        return FuncType([apply(p, subs) for p in type.params], apply(type.ret, subs))
    return type

def unify(type_a: Type, type_b: Type, **kwargs) -> dict[str, Type]:
    kwargs["env"] = kwargs["env"].copy()
    if isinstance(type_a, GenericVar):
        return {type_a.name: type_b}
    if isinstance(type_b, GenericVar):
        return {type_b.name: type_a}
    if isinstance(type_a, PlaceHolderFuncType):
        params, ret = type_a.params, type_a.ret
        assert isinstance(type_b, FuncType)
        param_types = type_b.params
        ret_type = type_b.ret
        for a, b in zip(params.children, param_types):
            kwargs["env"][a] = b
        new_ret_type, _ = typecheck(ret, **kwargs)
        assert len(params.children) == len(param_types)
        subs = unify(new_ret_type, ret_type, **kwargs)
        new_ret_type = apply(new_ret_type, subs, **kwargs)
        assert extends(new_ret_type, ret_type, **kwargs)
        return { repr(type_a): type_b }
    if isinstance(type_b, PlaceHolderFuncType):
        params, ret = type_b.params, type_b.ret
        assert isinstance(type_a, FuncType)
        param_types = type_a.params
        ret_type = type_a.ret
        for a, b in zip(params.children, param_types):
            kwargs["env"][a] = b
        new_ret_type, _ = typecheck(ret, **kwargs)
        assert len(params.children) == len(param_types)
        assert extends(new_ret_type, ret_type)
        return { repr(type_b): type_a }     
    if isinstance(type_a, PrimitiveType) and isinstance(type_b, PrimitiveType):
        return {}
    if isinstance(type_a, UnionType):
        subs = {}
        for v in type_a.values:
            subs.update(unify(v, type_b, **kwargs))
        return subs
    if isinstance(type_b, UnionType):
        subs = {}
        for v in type_b.values:
            subs.update(unify(type_a, v, **kwargs))
        return subs
    if isinstance(type_a, DictType) and isinstance(type_b, DictType):
        subs = {}
        subs.update(unify(type_a.key, type_b.key, **kwargs))
        subs.update(unify(type_a.value, type_b.value, **kwargs))
        return subs
    if isinstance(type_a, ObjType) and isinstance(type_b, ObjType):
        subs = {}
        for k, v in type_a.fields.items():
            if k in type_b.fields:
                subs.update(unify(v, type_b.fields[k], **kwargs))
        return subs
    if isinstance(type_a, MetaType) and isinstance(type_b, MetaType):
        subs = {}
        subs.update(unify(type_a.base, type_b.base, **kwargs))
        subs.update(unify(type_a.mt, type_b.mt, **kwargs))
        return subs
    if isinstance(type_a, FuncType) and isinstance(type_b, FuncType):
        subs = {}
        for a, b in zip(type_a.params, type_b.params):
            subs.update(unify(b, a, **kwargs))
        subs.update(unify(type_a.ret, type_b.ret, **kwargs))
        return subs
    return {}

def intersect(type_a: Type, type_b: Type, **kwargs) -> Type:
    if isinstance(type_a, PrimitiveType) and type_a.name == "never":
        return PrimitiveType("never")
    if isinstance(type_a, PrimitiveType) and type_a.name == "unknown":
        return simplify(type_b, **kwargs)
    if isinstance(type_b, PrimitiveType) and type_b.name == "never":
        return PrimitiveType("never")
    if isinstance(type_b, PrimitiveType) and type_b.name == "unknown":
        return simplify(type_a, **kwargs)
    if isinstance(type_a, PrimitiveType) and isinstance(type_b, PrimitiveType):
        if type_a.name == type_b.name:
            return simplify(type_a, **kwargs)
        return PrimitiveType("never")
    if isinstance(type_a, UnionType):
        return UnionType([intersect(v, type_b, **kwargs) for v in type_a.values]).simplify()
    if isinstance(type_b, UnionType):
        return UnionType([intersect(type_a, v, **kwargs) for v in type_b.values]).simplify()
    if isinstance(type_a, DictType) and isinstance(type_b, DictType):
        key = intersect(type_a.key, type_b.key, **kwargs)
        value = intersect(type_a.value, type_b.value, **kwargs)
        return DictType(key, value)
    if isinstance(type_a, ObjType) and isinstance(type_b, ObjType):
        obj_fields = {}
        for k, v in type_a.fields.items():
            obj_fields[k] = v
        for k, v in type_b.fields.items():
            if k in obj_fields:
                obj_fields[k] = intersect(obj_fields[k], v, **kwargs)
            else:
                obj_fields[k] = v
        return ObjType(obj_fields)
    if isinstance(type_a, MetaType) and isinstance(type_b, MetaType):
        base = intersect(type_a.base, type_b.base, **kwargs)
        mt = intersect(type_a.mt, type_b.mt, **kwargs)
        return MetaType(base, mt)
    if isinstance(type_a, FuncType) and isinstance(type_b, FuncType):
        params = [intersect(a, b, **kwargs) for a, b in zip(type_a.params, type_b.params)]
        ret = intersect(type_a.ret, type_b.ret, **kwargs)
        return FuncType(params, ret)
    return PrimitiveType("never")

def extends(type_a: Type, type_b: Type, **kwargs) -> bool:
    result, _ = extends_with_err(type_a, type_b, **kwargs)
    return result

def extends_with_err(type_a: Type, type_b: Type, **kwargs) -> tuple[bool, Optional[str]]:
    if isinstance(type_a, PlaceHolderType) and isinstance(type_b, PlaceHolderType):
        if type_a.name != type_b.name:
            return False, f"Type '{type_a}' does not extend '{type_b}'"
        return True, None
    type_a = eval_type(kwargs["type_env"], type_a)
    type_b = eval_type(kwargs["type_env"], type_b)
    if kwargs.get("allow_generics") and isinstance(type_b, GenericVar):
        return True, None
    if isinstance(type_a, PrimitiveType) and type_a.name == "never":
        return True, None
    if isinstance(type_b, PrimitiveType) and type_b.name == "unknown":
        return True, None
    if isinstance(type_a, PrimitiveType) and isinstance(type_b, PrimitiveType):
        if type_a.name == "integer" and type_b.name == "number":
            return True, None
        if type_a.name == type_b.name:
            return True, None
        return False, f"Type '{type_a}' does not extend '{type_b}'"
    if isinstance(type_a, UnionType):
        msg = None
        result = True
        for v in type_a.values:
            b, m = extends_with_err(v, type_b, **kwargs)
            if not b:
                msg = m
                result = False
                break
        if not result:
            return result, f"Union type '{type_a}' does not extend '{type_b}'\n-> {msg}"
        return result, None
    if isinstance(type_b, UnionType):
        msg = None
        result = False
        for v in type_b.values:
            b, m = extends_with_err(type_a, v, **kwargs)
            if b:
                result = True
                break
            msg = m
        if not result:
            return result, f"Type '{type_a}' does not extend '{type_b}'\n-> {msg}"
        return result, None
    if isinstance(type_a, DictType) and isinstance(type_b, DictType):
        key_result, key_msg = extends_with_err(type_a.key, type_b.key, **kwargs)
        if not key_result:
            return False, f"Key '{type_a.key}' does not extend '{type_b.key}'\n-> {key_msg}"
        value_result, value_msg = extends_with_err(type_a.value, type_b.value, **kwargs)
        if not value_result:
            return False, value_msg
        return True, None
    if isinstance(type_a, ObjType) and isinstance(type_b, ObjType):
        for k, v in type_a.fields.items():
            if k in type_b.fields:
                result, msg = extends_with_err(v, type_b.fields[k], **kwargs)
                if not result:
                    return False, msg
            if k not in type_b.fields:
                return False, f"Key '{k}' not found in '{type_b}'"
        return True, None
    if isinstance(type_a, MetaType) and isinstance(type_b, MetaType):
        base_result, base_msg = extends_with_err(type_a.base, type_b.base, **kwargs)
        if not base_result:
            return False, f"Base '{type_a.base}' does not extend '{type_b.base}'\n-> {base_msg}"
        mt_result, mt_msg = extends_with_err(type_a.mt, type_b.mt, **kwargs)
        if not mt_result:
            return False, f"Meta '{type_a.mt}' does not extend '{type_b.mt}'\n-> {mt_msg}"
        return True, None
    if isinstance(type_a, FuncType) and isinstance(type_b, FuncType):
        if len(type_a.params) != len(type_b.params):
            return False, f"Function type '{type_a}' does not have the same number of parameters as '{type_b}'"
        ret_result, ret_msg = extends_with_err(type_a.ret, type_b.ret, **kwargs)
        if not ret_result:
            return False, f"Return type '{type_a.ret}' does not extend '{type_b.ret}'\n-> {ret_msg}"       
        for a, b in zip(type_a.params, type_b.params):
            param_result, param_msg = extends_with_err(a, b, **kwargs)
            if not param_result:
                return False, f"Parameter '{a}' does not extend '{b}'\n-> {param_msg}"
        return True, None
    return False, f"Type '{type_a}' does not extend '{type_b}'"

def typecheck_integer(value, **kwargs):
    return PrimitiveType("integer", value), False

def typecheck_number(value, **kwargs):
    return PrimitiveType("number", value), False

def typecheck_string(value, **kwargs):
    return PrimitiveType("string", value), False

def typecheck_boolean(value, **kwargs):
    return PrimitiveType("boolean", value), False

def typecheck_nil(value, **kwargs):
    return PrimitiveType("nil", value), False

def typecheck_name(value, **kwargs):
    env = kwargs["env"]
    if str(value) in env:
        return env[value], False
    type_error(get_loc(value), f"Accessing unbound variable '{value}'")
    assert False

def typecheck_identifier(value, **kwargs):
    return PrimitiveType("string"), False

def typecheck_type_name(value, **kwargs):
    type_env = kwargs["type_env"]
    if value in type_env:
        return type_env[value], False
    return PlaceHolderType(value), False

def typecheck_generic_type(name, *generics, **kwargs):
    if name not in kwargs["type_env"]:
        type_error(get_loc(name), f"Attempt to reference an undefined type '{name}'")
    type = kwargs["type_env"][name]
    subs = {}
    for generic, value in zip(type.generics, generics):
        value, _ = typecheck(value, **kwargs)
        subs[generic] = value
    return apply(type.type, subs, **kwargs), False

def typecheck_unary_expr(op: Token, value: Node, **kwargs):
    value_type = typecheck(value, **kwargs)
    assert isinstance(value_type, PrimitiveType) # TODO: add support for other types
    match op:
        case "-":
            if value_type.name == "integer":
                return PrimitiveType("integer"), False
            if value_type.name == "number":
                return PrimitiveType("number"), False
            type_error(get_loc(op), f"Attempt to perform arithmetic (unary -) on a non-numeric value")
        case "#":
            # TODO: add support for length of tables
            if value_type.name == "string":
                return PrimitiveType("integer"), False
            type_error(get_loc(op), f"Attempt to obtain length of a non-string value")
        case "not":
            if value_type.name == "boolean":
                return PrimitiveType("boolean"), False
            type_error(get_loc(op), f"Attempt to perform logical (not) on a non-boolean value")
    assert False, f"Unknown unary operator {op}"

def typecheck_binary_expr(left: Node, op: Token, right: Node, **kwargs):
    left_type, _ = typecheck(left, **kwargs)
    right_type, _ = typecheck(right, **kwargs)
    match str(op):
        case x if x in "+-*/^%":
            assert isinstance(left_type, PrimitiveType)
            assert isinstance(right_type, PrimitiveType)
            if left_type.name == "integer" and right_type.name == "integer":
                return PrimitiveType("integer"), False
            if left_type.name == "number" and right_type.name == "number":
                return PrimitiveType("number"), False
            type_error(get_loc(op), f"Attempt to perform arithmetic ({op}) on non-numeric values: '{left_type}' and '{right_type}'")
        case "..":
            assert isinstance(left_type, PrimitiveType)
            assert isinstance(right_type, PrimitiveType)
            if left_type.name == "string" and right_type.name == "string":
                return PrimitiveType("string"), False
            type_error(get_loc(op), f"Attempt to concatenate non-string values: '{left_type}' and '{right_type}'")
        case x if x in ["==", "~=", "<", "<=", ">", ">="]:
            if extends(left_type, right_type, **kwargs):
                return PrimitiveType("boolean"), False
            type_error(get_loc(op), f"Attempt to compare values of different types: '{left_type}' and '{right_type}'")
        case x if x in ["and", "or"]:
            return UnionType([left_type, right_type]), False
    assert False, f"Unknown binary operator {op}"

def typecheck_array(fields: Tree, **kwargs):
    last_field = PrimitiveType("never")
    for field in fields.children:
        if len(field.children) == 2:
            _, value = field.children
        else: value = field.children[0]
        field_type, _ = typecheck(value, **kwargs)
        if not extends(field_type, last_field, **kwargs):
            type_error(get_loc(value), f"Table contains non-homogeneous value types: '{field_type}' and '{last_field}'")
    return DictType(PrimitiveType("integer"), last_field), False

def typecheck_obj(fields: Tree, **kwargs):
    obj_fields = {}
    for field in fields.children:
        key, value = field.children
        value_type, _ = typecheck(value, **kwargs)
        obj_fields[str(key)] = value_type
    return ObjType(obj_fields), False

def typecheck_dict(fields: Tree, **kwargs):
    last_key_type = PrimitiveType("never")
    last_value_type = PrimitiveType("never")
    for field in fields.children:
        key, value = field.children
        key_type, _ = typecheck(key, **kwargs)
        value_type, _ = typecheck(value, **kwargs)
        if not extends(last_key_type, key_type, **kwargs):
            type_error(get_loc(key), f"Attempt to create a table with different key types: '{key_type}' and '{last_key_type}'")
        if not extends(last_value_type, value_type, **kwargs):
            type_error(get_loc(value), f"Attempt to create a table with non-homogeneous value types")
        last_key_type, last_value_type = key_type, value_type
    return DictType(key_type, value_type), False

def typecheck_table(fields: Tree, **kwargs):
    is_array = True
    is_obj = True
    for field in fields.children:
        if len(field.children) == 2:
            key, _ = field.children
            key_type, _ = typecheck(key, **kwargs)
            if not isinstance(key_type, PrimitiveType) or key_type.name != "integer":
                is_array = False
            if not isinstance(key_type, PrimitiveType) or key_type.name == "string":
                is_obj = True
    if is_array: return typecheck_array(fields, **kwargs)
    if is_obj: return typecheck_obj(fields, **kwargs)
    return typecheck_dict(fields, **kwargs)

def typecheck_prop_expr(obj, prop, **kwargs):
    obj_type, _ = typecheck(obj, **kwargs)
    obj_type = eval_type(kwargs["type_env"], obj_type)
    if not isinstance(obj_type, ObjType):
        type_error(get_loc(obj), f"Attempt to access a property of a non-object value")
        assert False
    if str(prop) not in obj_type.fields:
        type_error(get_loc(prop), f"Attempt to access a non-existent property '{prop}'")
    return obj_type.fields[str(prop)], False

def typecheck_alias_type(name, generics, type_expr, **kwargs):
    type_env = kwargs["type_env"]
    if len(generics.children) == 1 and generics.children[0].data == "empty":
        generics = []
    else:
        generics = [str(g.children[0]) for g in generics.children]
    if name in type_env:
        type_error(get_loc(name), f"Attempt to redefine type '{name}'")
    if generics:
        type_env[str(name)] = GenericAlias(typecheck(type_expr, **kwargs)[0], generics)
    else:
        type_env[str(name)] = typecheck(type_expr, **kwargs)[0]
    return type_env[str(name)], False

def typecheck_meta_type(base, mt, **kwargs):
    base_type, _ = typecheck(base, **kwargs)
    mt_type, _ = typecheck(mt, **kwargs)
    return MetaType(base_type, mt_type), False

def typecheck_func_type(*args, **kwargs):
    generics, params, ret = args
    if len(generics.children) == 1 and generics.children[0].data == "empty":
        generics = []
    else:
        generics = [str(g.children[0]) for g in generics.children]
    kwargs["type_env"] = kwargs["type_env"].copy()
    for g in generics:
        kwargs["type_env"][g] = GenericVar(g)
    params = [typecheck(param, **kwargs)[0] for param in params.children]
    ret = typecheck(ret, **kwargs)[0]
    return FuncType(params, ret, generics), False

def typecheck_type_hint(type_expr, **kwargs):
    return typecheck(type_expr, **kwargs)

def typecheck_union_type(left, right, **kwargs):
    return UnionType([typecheck(left, **kwargs)[0], typecheck(right, **kwargs)[0]]), False

def typecheck_primitive_type(name, **kwargs):
    return PrimitiveType(name), False

def typecheck_obj_type(*fields, **kwargs):
    obj_fields = {}
    for field in fields:
        key, value = field.children
        value_type, _ = typecheck(value, **kwargs)
        obj_fields[str(key)] = value_type
    return ObjType(obj_fields), False

def typecheck_return_stmt(*values, **kwargs):
    if len(values) == 0:
        return PrimitiveType("nil"), True
    value = values[0]
    return typecheck(value, **kwargs)[0], True

def typecheck_func_call(name, args, **kwargs):
    name_loc = get_loc(name)
    name, _ = typecheck(name, **kwargs)
    arg_locs = [get_loc(arg) for arg in args.children]
    args = [typecheck(arg, **kwargs)[0] for arg in args.children]
    if isinstance(name, PlaceHolderFuncType):
        type_error(name_loc, f"Attempting to call an un-typed function")
        assert False
    if not isinstance(name, FuncType):
        type_error(name_loc, f"Attempting to call a non-function value")
        assert False
    if len(args) != len(name.params):
        type_error(name_loc, f"Attempting to call a function that requires {len(name.params)} parameters with {len(args)} arguments")
    i = 0
    subs = {}
    kwargs["allow_generics"] = True
    for arg, param in zip(args, name.params):
        param = apply(param, subs, **kwargs)
        if not extends(arg, param, **kwargs):
            type_error(arg_locs[i], f"Invalid argument type, expected '{param}', got '{arg}'")
        subs.update(unify(arg, param, **kwargs))
        i += 1
    return apply(name.ret, subs, **kwargs), False

def typecheck_method_call(obj, name, args, **kwargs):
    func = Tree("prop_expr", [obj, name])
    new_args = Tree("args", [obj] + args.children)
    return typecheck_func_call(func, new_args, **kwargs)

def typecheck_var_decl(*args, **kwargs):
    if len(args) == 3:
        type_hint, var, expr = args
    else:
        type_hint = None
        var, expr = args
    assert isinstance(var, Token) # TODO: add support for other types of prefix exprs
    kwargs["allow_generics"] = True
    expr_type, _ = typecheck(expr, **kwargs)
    if type_hint is not None:
        type_hint, _ = typecheck(type_hint, **kwargs)
        subs = unify(expr_type, type_hint, **kwargs)
        new_expr_type = apply(expr_type, subs, **kwargs)
        result, msg = extends_with_err(new_expr_type, type_hint, **kwargs)
        if not result:
            type_error(get_loc(expr), f"Invalid assignment, expected '{type_hint}', got '{new_expr_type}'\n-> {msg}")
        kwargs["env"][str(var)] = type_hint
    else:
        kwargs["env"][str(var)] = expr_type
    return PrimitiveType("nil"), False

def typecheck_func_body(params, types, body, **kwargs):
    params = params.children[0]
    new_params = []
    generics: list[str] = []
    i = 0
    if len(types.children) == 0:
        return PlaceHolderFuncType(params, body), False
    for type in types.children:
        if type.data == "generic_type_hint":
            generics = list(map(lambda x: str(x.children[0]), type.children))
            kwargs["type_env"].update({g: GenericVar(g) for g in generics})
            continue
        param = params.children[i]
        pname, ptype = type.children
        assert param.type == "NAME" and param.value == pname.value
        p, _ = typecheck(ptype, **kwargs)
        new_params.append(p)
        kwargs["env"][str(param)] = p
        i += 1
    body_type, body_ret = typecheck(body, **kwargs)
    return FuncType(new_params, body_type, generics), body_ret

def typecheck_func_expr(body, **kwargs):
    return typecheck(body, **kwargs)

def typecheck_func_decl(name, body, **kwargs):
    assert name.data == "func_name"
    name = name.children[0]
    body_type, _ = typecheck(body, **kwargs)
    kwargs["env"][str(name)] = body_type
    return PrimitiveType("nil"), False

def _update_type_by_check(condition, **kwargs):
    if isinstance(condition, Tree) and condition.data == "eq_expr":
        left, op, right = condition.children
        assert isinstance(op, Token)
        if op.value == "==":
            if isinstance(left, Token) and kwargs["env"].get(left.value) is not None:
                kwargs["env"][left.value], _ = typecheck(right, env=kwargs["env"])
            if isinstance(right, Token) and kwargs["env"].get(right.value) is not None:
                kwargs["env"][right.value], _ = typecheck(left, env=kwargs["env"])
            if isinstance(left, Tree) and left.data == "func_call":
                func, args = left.children
                if isinstance(func, Token) and func.value == "type" and len(args.children) == 1:
                    left_var = args.children[0]
                    if isinstance(left_var, Token) and kwargs["env"].get(left_var.value) is not None:
                        if isinstance(right, Token) and right.type == "STRING":
                            left_type = kwargs["env"][left_var.value]
                            r = right.value[1:-1]
                            if r == "integer":
                                l = intersect(
                                    left_type,
                                    PrimitiveType("integer"))
                            if r == "number":
                                l = intersect(
                                    left_type,
                                    PrimitiveType("number"))
                            if r == "string":
                                l = intersect(
                                    left_type,
                                    PrimitiveType("string"))
                            if r == "boolean":
                                l = intersect(
                                    left_type,
                                    PrimitiveType("boolean"))
                            if r == "nil":
                                l = intersect(
                                    left_type,
                                    PrimitiveType("nil"))
                            if r == "table":
                                l = intersect(
                                    left_type,
                                    DictType(PrimitiveType("unknown"), PrimitiveType("unknown")))
                            if r == "function":
                                l = intersect(
                                    left_type,
                                    FuncType([
                                        PrimitiveType("unknown")
                                    ], PrimitiveType("unknown")))
                            if isinstance(l, PrimitiveType) and l.name == "never":
                                type_error(get_loc(left), f"Attempting to compare two distinct types '{left_type}', and '{r}'")
                            kwargs["env"][left_var.value] = l

def typecheck_if_stmt(cond, body, **kwargs):
    kwargs["env"] = kwargs["env"].copy()
    _update_type_by_check(cond, **kwargs)
    cond_type, _ = typecheck(cond, **kwargs)
    return typecheck(body, **kwargs)

def typecheck_chunk(*stmts, **kwargs):
    new_env = kwargs["env"].copy()
    last = None
    rets = False
    kwargs["env"] = new_env
    for stmt in stmts:
        type, ret = typecheck(stmt, **kwargs)
        if ret:
            if last is None:
                last = type
            else:
                last = UnionType([last, type])
            rets = True
    if last is None:
        return PrimitiveType("nil"), rets
    return last, rets

DATA_TO_TYPECHECKER: dict[str, Callable[..., tuple[Type, bool]]] = {
    "INTEGER": typecheck_integer,
    "NUMBER": typecheck_number,
    "STRING": typecheck_string,
    "BOOLEAN": typecheck_boolean,
    "NIL": typecheck_nil,
    "NAME": typecheck_name,
    "IDENTIFIER": typecheck_identifier,
    "TYPE_NAME": typecheck_type_name,
    "unary_expr": typecheck_unary_expr,
    "exp_expr": typecheck_binary_expr,
    "mul_expr": typecheck_binary_expr,
    "add_expr": typecheck_binary_expr,
    "rel_expr": typecheck_binary_expr,
    "eq_expr": typecheck_binary_expr,
    "log_expr": typecheck_binary_expr,
    "table": typecheck_table,
    "prop_expr": typecheck_prop_expr,
    "alias_type": typecheck_alias_type,
    "meta_type": typecheck_meta_type,
    "generic_type": typecheck_generic_type,
    "func_type": typecheck_func_type,
    "type_hint": typecheck_type_hint,
    "union_type": typecheck_union_type,
    "PRIMITIVE_TYPE": typecheck_primitive_type,
    "obj_type": typecheck_obj_type,
    "return_stmt": typecheck_return_stmt,
    "func_call": typecheck_func_call,
    "method_call": typecheck_method_call,
    "var_decl": typecheck_var_decl,
    "if_stmt": typecheck_if_stmt,
    "func_body": typecheck_func_body,
    "func_expr": typecheck_func_expr,
    "func_decl": typecheck_func_decl,
    "chunk": typecheck_chunk,
}

def typecheck(tree: Node, **kwargs):
    if isinstance(tree, Token):
        type, ret = DATA_TO_TYPECHECKER[tree.type](tree, **kwargs)
        return simplify(type, **kwargs), ret
    type, ret = DATA_TO_TYPECHECKER[tree.data](*tree.children, **kwargs)
    return simplify(type, **kwargs), ret

_LUAENV = {
    "print": FuncType([PrimitiveType("unknown")], PrimitiveType("nil")),
    "type": FuncType([PrimitiveType("unknown")], PrimitiveType("string")),
}

T_1 = GenericVar("T")
M_1 = GenericVar("M")
_LUAENV["setmetatable"] = FuncType([T_1, M_1], MetaType(T_1, M_1), ["T", "M"])

def main():
    type_env = {}
    env = _LUAENV
    with open("test.lua") as f:
        tree = parser.parse(f.read())
        program, returns = typecheck(tree, env=env, type_env=type_env)
        print(program)

if __name__ == "__main__":
    main()
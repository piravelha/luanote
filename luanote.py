from re import A
from typing import Any, Callable
from lark import Lark, Token, Tree

EXPAND_ALIASES = False

with open("grammar.lark") as f:
    grammar = f.read()

parser = Lark(grammar)

class Type:
    def __init__(self, name: str, generics: 'list[Type]' = []):
        self.name = name
        self.alias = None
        self.generics = generics
        self.args = []
    def set_alias(self, alias: str, generics: 'list[Type]'):
        self.generics = generics
        self.alias = alias
    def from_type(self, type: 'Type'):
        self.alias = type.alias
        self.args = type.args
        return self
    def copy(self):
        return Type(self.name, self.generics)
    def alias_repr(self):
        if self.args:
            args = ", ".join([repr(a) for a in self.args])
            return f"{self.alias}<{args}>"
        return self.alias or ""
    def repr(self, alias: str | None = None):
        if self.alias and not EXPAND_ALIASES:
            return self.alias_repr()
        if alias is not None and self.alias == alias:
            return self.alias_repr()
        alias = alias or self.alias
        if self.generics:
            gens = ", ".join([g.repr(alias) for g in self.generics])
            return f"{self.name}<{gens}>"
        return self.name
    def __repr__(self):
        return self.repr(None)

class TableType(Type):
    def __init__(self,
            name: str,
            key: Type,
            value: Type,
            fields: dict[str | int, Callable[[], Type]] | None = None,
            meta: Type | None = None):
        super().__init__(name)
        self.key = key
        self.value = value
        self.fields = fields
        self.meta = meta
    def copy(self):
        return TableType(self.name, self.key, self.value, self.fields, self.meta)
    def repr(self, alias: str | None = None):
        if self.alias and not EXPAND_ALIASES:
            return self.alias_repr()
        if alias is not None and self.alias == alias:
            return self.alias_repr()
        alias = alias or self.alias
        if self.key.name == "integer":
            return f"{self.value}[]"
        if self.fields and len(self.fields) > 0:
            return f"{{ {', '.join(f'{k}: {v().repr(alias)}' for k, v in self.fields.items())} }}"
        return f"{{ [{self.key.repr(alias)}]: {self.value.repr(alias)} }}"

class MetaType(TableType):
    def __init__(self, base: Type, meta: Type):
        super().__init__("meta", base.key if isinstance(base, TableType) else Type("any"),  base.value if isinstance(base, TableType) else Type("any"), base.fields if isinstance(base, TableType) else None, meta)
        self.base = base
        self.meta = meta
    def copy(self):
        return MetaType(self.base, self.meta)
    def repr(self, alias: str | None = None):
        if self.alias and not EXPAND_ALIASES:
            return self.alias_repr()
        if alias is not None and self.alias == alias:
            return self.alias_repr()
        alias = alias or self.alias
        return f"{self.base.repr(alias)} @meta {self.meta.repr(alias)}"

class FunctionType(Type):
    def __init__(self, params: list[Type], returns: Type):
        super().__init__("function")
        self.params = params
        self.returns = returns
    def copy(self):
        return FunctionType(self.params, self.returns)
    def repr(self, alias: str | None = None):
        if self.alias and not EXPAND_ALIASES:
            return self.alias_repr()
        if alias is not None and self.alias == alias:
            return self.alias_repr()
        alias = alias or self.alias
        return f"({', '.join(map(lambda p: p.repr(alias), self.params))}) -> {self.returns.repr(alias)}"

class UnionType(Type):
    def __init__(self, name: str, *values: Type):
        super().__init__(name)
        self.values = values
    def copy(self):
        return UnionType(self.name, *self.values)
    def repr(self, alias: str | None):
        if self.alias and not EXPAND_ALIASES:
            return self.alias_repr()
        if alias is not None and self.alias == alias:
            return self.alias_repr()
        alias = alias or self.alias
        return " | ".join([v.repr(alias) for v in self.values])

class GenericType(Type):
    _iota = 0
    def __init__(self, gname: str, constraint: Type = Type("any")):
        super().__init__("generic")
        self.gname = gname
        self.gconstraint = constraint
        self.iota = GenericType._iota
        GenericType._iota += 1
    def copy(self):
        return GenericType(self.gname)
    def repr(self, alias: str | None = None):
        if self.gconstraint.name != "any":
            return f"{self.gname} : {self.gconstraint.repr(alias)}"
        return f"{self.gname}"

_LUAENV: dict[str, Callable[[], Type]] = {
    "print": lambda: FunctionType([Type("any")], Type("nil")),
    "tostring": lambda: FunctionType([Type("any")], Type("string")),
}

T_1 = GenericType("T", TableType("table", Type("any"), Type("any")))
U_1 = GenericType("U", TableType("table", Type("any"), Type("any")))
_LUAENV["setmetatable"] = lambda: FunctionType([T_1, U_1], MetaType(T_1, U_1))

def apply(type: Type, subs: dict[str, Type]) -> Type:
    old = type
    args = []
    for a in type.args:
        args.append(apply(a, subs))
    new = type.copy().from_type(type)
    new.args = args
    type = new
    if isinstance(type, GenericType):
        if subs.get(type.gname):
            return subs[type.gname]
        return old
    if isinstance(type, UnionType):
        new = []
        for x in type.values:
            new.append(apply(x, subs))
        return UnionType("union", *new).from_type(type)
    if isinstance(type, MetaType):
        base = apply(type.base, subs)
        meta = apply(type.meta, subs)
        return MetaType(base, meta).from_type(type)
    if isinstance(type, TableType):
        if type.fields is not None:
            fields = {}
            for k, v in type.fields.items():
                fields[k] = lambda: apply(v(), subs)
            return TableType("table", type.key, type.value, fields).from_type(type)
        key = apply(type.key, subs)
        value = apply(type.value, subs)
        return TableType("table", key, value).from_type(type)
    if isinstance(type, FunctionType):
        params = []
        for p in type.params:
            params.append(apply(p, subs))
        return FunctionType(params, apply(type.returns, subs)).from_type(type)
    return type

def unify(type_a: Type, type_b: Type, loc: str = "???:?:?") -> dict[str, Type]:
    if isinstance(type_b, GenericType):
        if not extends(type_a, type_b.gconstraint):
            raise ValueError(f"{loc} Generic argument '{type_a}' does not follow its generic constraint '{type_b}'")
        return {
            type_b.gname: type_a
        }
    if isinstance(type_a, UnionType) and isinstance(type_b, UnionType):
        subs = {}
        for a, b in zip(type_a.values, type_b.values):
            subs.update(unify(a, b, loc))
        return subs
    if isinstance(type_a, MetaType) and isinstance(type_b, MetaType):
        subs = {}
        subs.update(unify(type_a.base, type_b.base, loc))
        subs.update(unify(type_a.meta, type_b.meta, loc))
        return subs
    if isinstance(type_a, TableType) and isinstance(type_b, TableType):
        if type_a.fields is not None and type_b.fields is not None:
            subs: dict[str, Type] = {}
            for k, a in type_a.fields.items():
                b = type_b.fields[k]
                new: dict[str, Type] = unify(a(), b(), loc)
                subs.update(new)
            return subs
        subs = {}
        subs.update(unify(type_a.key, type_b.key, loc))
        subs.update(unify(type_a.value, type_b.value, loc))
        return subs
    if isinstance(type_a, FunctionType) and isinstance(type_b, FunctionType):
        subs = {}
        for a, b in zip(type_a.params, type_b.params):
            subs.update(unify(a, b, loc))
        subs.update(unify(type_a.returns, type_b.returns, loc))
        return subs
    return {}
            
    
def extends(type_a: Type, type_b: Type, cur_aliases: list[str | None] = []) -> bool:
    if type_a.alias is not None and type_b.alias is not None and (type_a.alias in cur_aliases or type_b.alias in cur_aliases):
        return True
    if type_b.name == "generic":
        return True
    if type_b.name == "any":
        return True
    if type_a.name == "never":
        return True
    if isinstance(type_a, UnionType):
        for x in type_a.values:
            if not extends(x, type_b, cur_aliases + [x.alias]):
                return False
        return True
    if isinstance(type_b, UnionType):
        for x in type_b.values:
            if extends(type_a, x, cur_aliases + [x.alias]):
                return True
        return False
    if type_a.name == "integer" and type_b.name == "number":
        return True
    if isinstance(type_a, MetaType) and isinstance(type_b, MetaType):
        if not extends(type_a.base, type_b.base, cur_aliases + [type_a.base.alias, type_b.base.alias]):
            return False
        if not extends(type_a.meta, type_b.meta, cur_aliases + [type_a.meta.alias, type_b.meta.alias]):
            return False
        return True
    if isinstance(type_a, TableType) and isinstance(type_b, TableType):
        if type_a.fields is not None and type_b.fields is not None:
            for key, value in type_b.fields.items():
                if key not in type_a.fields:
                    value = value()
                    return extends(Type("nil"), value, cur_aliases + [value.alias])
                value = value()
                if not extends(type_a.fields[key](), value, cur_aliases + [value.alias]):
                    return False
            return True
        if not extends(type_a.key, type_b.key, cur_aliases + [type_a.key.alias, type_b.key.alias]):
            return False
        if not extends(type_a.value, type_b.value, cur_aliases + [type_a.value.alias, type_b.value.alias]):
            return False
    if isinstance(type_a, FunctionType) and isinstance(type_b, FunctionType):
        if len(type_a.params) != len(type_b.params):
            return False
        for a, b in zip(type_a.params, type_b.params):
            if not extends(b, a, cur_aliases + [a.alias, b.alias]):
                return False
        if not extends(type_a.returns, type_b.returns, cur_aliases + [type_a.returns.alias, type_b.returns.alias]):
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

def get_index_path(expr: Tree, env: dict[str, Callable[[], Type]], typeenv) -> Type:
    def get_base(expr: Tree | Token):
        if isinstance(expr, Token):
            return expr.value
        base, _ = expr.children
        return get_base(base)
    base = env[get_base(expr)]
    def run(expr: Tree, path: Any) -> Any:
        if isinstance(expr, Token):
            return path
        if expr.data == "index_expr":
            tbl, key = expr.children
            path = run(tbl, path)
            key = infer(key, env, typeenv)
            type = TableType("table", key, Type("any"))
            if not extends(path, type):
                assert False
            return run(tbl, path.value)
        if expr.data == "prop_expr":
            tbl, prop = expr.children
            path = run(tbl, path)
            if path.fields is not None:
                if not path.fields.get(str(prop)):
                    assert False
                return path.fields[str(prop)]()
            type = TableType("table", Type("string"), Type("any"))
            if not extends(path, type):
                assert False
            return path.value
        assert False
    return run(expr, base)

def infer(tree: Tree | Token, env: dict[str, Callable[[], Type]] = _LUAENV, typeenv: dict[str, Callable[[], Type]] = {}) -> Type:
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
            type = env.get(tree.value)
            return env[tree.value]()
        if tree.type == "PRIMITIVE_TYPE":
            return Type(tree.value)
        if tree.type == "TYPE_NAME":
            type = typeenv[tree.value]()
            if typeenv.get(tree.value + "@cached"):
                return typeenv[tree.value + "@cached"]()
            typeenv[tree.value + "@cached"] = lambda: type
            return type
        assert False, f"Not implemented: {tree.type}"
    if tree.data == "unary_expr":
        op, value = tree.children
        value = infer(value, env, typeenv)
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
        left = infer(left, env, typeenv)
        right = infer(right, env, typeenv)
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
        fields2 = {}
        cur_index = 1
        field_list = tree.children[0]
        new_array = []
        is_array = True
        different_keys = False
        different_values = False
        key_type = None
        value_type = None
        last_key = None
        last_value = None
        def run(field):
            nonlocal cur_index, is_array, new_array, different_keys, different_values, key_type, value_type, last_key, last_value, fields2
            if isinstance(field, Tree) and len(field.children) == 2:
                is_array = False
                key, value = field.children
                if isinstance(key, Tree):
                    cur_key_type = infer(key, env, typeenv)
                else:
                    cur_key_type = Type("string")
                if key_type is None:
                    key_type = cur_key_type
                if not extends(cur_key_type, key_type) and not extends(key_type, cur_key_type):
                    different_keys = True
                value = infer(value, env, typeenv)
                if value_type is None:
                    value_type = value
                if not extends(value, value_type, [value.alias, value_type.alias]) and not extends(value_type, value, [value.alias, value_type.alias]):
                    different_values = True
                fields2[str(key)] = lambda: value
                return None
            field = infer(field, env, typeenv)
            fields2[cur_index] = field
            new_array.append(field)
            cur_index = cur_index + 1
        for field in field_list.children:
            run(field)
        if is_array:
            if different_keys:
                raise ValueError(f"{loc(tree)} Table contains keys of different types: '{key_type}' and '{last_key}'")
            if different_values:
                raise ValueError(f"{loc(tree)} Table contains values of different types: '{value_type}' and '{last_value}'")
            if not new_array:
                return TableType("table", Type("never"), Type("never"))
            first = new_array[0]
            for field in new_array:
                if not extends(field, first) and not extends(first, field):
                    raise ValueError(f"{loc(tree)} Array contains elements of different types: '{first}' and '{field}'")
            return TableType("table", Type("integer"), first)
        assert key_type is not None and value_type is not None
        return TableType("table", key_type, value_type,  fields2)
    if tree.data == "index_expr":
        table, index = tree.children
        index_loc = loc(index)
        table = infer(table, env, typeenv)
        index = infer(index, env, typeenv)
        if isinstance(table, TableType):
            if extends(index, table.key):
                return table.value
            raise ValueError(f"{index_loc} Invalid index type '{index}' for table with key type '{table.key}'")
        raise ValueError(f"{index_loc} Attempting to index a non-table type: '{table}'")
    if tree.data == "prop_expr":
        table, prop = tree.children
        prop_loc = loc(prop)
        table = infer(table, env, typeenv)
        assert isinstance(prop, Token)
        prop = prop.value
        if isinstance(table, TableType):
            if table.fields is not None:
                if prop in table.fields:
                    return table.fields[prop]()
                raise ValueError(f"{prop_loc} Table '{table}' does not contain property '{prop}'")
            if table.key.name == "string":
                return table.value
            raise ValueError(f"{prop_loc} Invalid property type '{prop}' for table with value type '{table.value}'")
        raise ValueError(f"{prop_loc} Attempting to access a property of a non-table type: '{table}'")
    if tree.data == "assign_stmt":
        prefix, value = tree.children
        value_loc = loc(value)
        type = get_index_path(prefix, env, typeenv)
        value = infer(value, env, typeenv)
        if not extends(value, type):
            raise ValueError(f"{value_loc} Attempting to assign a value of type '{value}' to a variable of type '{type}'")
        return Type("nil")
    if tree.data == "func_call":
        prefix, args = tree.children
        prefix_loc = loc(prefix)
        prefix = infer(prefix, env, typeenv)
        if isinstance(args, Token):
            args = [infer(args, env, typeenv)]
        elif args.data == "table":
            args = [infer(args, env, typeenv)]
        else:
            args = [infer(arg, env, typeenv) for arg in args.children]
        if isinstance(prefix, FunctionType):
            if len(args) != len(prefix.params):
                raise ValueError(f"{prefix_loc} Invalid number of arguments")
            subs = {}
            for _, (arg, param) in enumerate(zip(args, prefix.params)):
                param = apply(param, subs)
                subs.update(unify(arg, param, prefix_loc))
                if not extends(arg, param):
                    raise ValueError(f"{prefix_loc} Invalid argument type, expected '{param}' but got '{arg}'")
            return apply(prefix.returns, subs)
        raise ValueError(f"{prefix_loc} Attempting to call a non-function type: '{prefix}'")
    if tree.data == "func_body":
        if len(tree.children) == 1:
            params = Tree("param_list", [])
            body = tree.children[0]
        else:
            params, body = tree.children
        new_env = env.copy()
        for param in params.children:
            new_env[str(param)] = lambda: Type("any")
        return FunctionType([Type("any") for _ in params.children], infer(body, new_env, typeenv))
    if tree.data == "func_expr":
        return infer(tree.children[0], env, typeenv)
    if tree.data == "var_decl":
        # TODO: support multiple var decls together
        if len(tree.children) == 3:
            type, varlist, exprlist = tree.children
            type = infer(type, env, typeenv)
            var, expr = varlist.children[0], exprlist.children[0]
            var_loc = loc(var)
            expr = infer(expr, env, typeenv)
            if not extends(expr, type):
                raise ValueError(f"{var_loc} Variable does not match its declared type, expected '{type}', got {expr}'")
            env[str(var)] = lambda: type
            return Type("nil")
        varlist, exprlist = tree.children
        var, expr = varlist.children[0], exprlist.children[0]
        env[str(var)] = lambda: infer(expr, env, typeenv)
        return Type("nil")
    if tree.data == "func_decl":
        # TODO: add support for variadic functions
        type, name, body = tree.children
        new_env = env.copy()
        new_typeenv = typeenv.copy()
        func_params, func_body = body.children
        func_params = func_params.children[0]
        expect_return = None
        for param in func_params.children:
            new_env[str(param)] = lambda: Type("any")
        for param in type.children:
            if param.data == "generic_type_hint":
                for p in param.children:
                    if len(p.children) == 1:
                        p = p.children[0]
                        new_typeenv[str(p)] = lambda: GenericType(str(p), Type("any"))
                        continue
                    pname, constraint = p.children
                    @lambda _: _(constraint)
                    def run_(constraint):
                        constraint = infer(constraint, env, new_typeenv)
                        new_typeenv[str(pname)] = lambda: GenericType(str(pname), constraint)
                continue
            if param.data == "return_type_hint":
                expect_return = infer(param.children[0], env, new_typeenv)
                continue
                
            pname, ptype = param.children
            new_ptype = infer(ptype, env, new_typeenv)
            new_env[str(pname)] = lambda: new_ptype
        str_name = ""
        if len(name.children) == 1:
            str_name = name.children[0]
            ret = infer(func_body, new_env, new_typeenv)
            if expect_return is not None:
                if not extends(ret, expect_return):
                    raise ValueError(f"{loc(tree)} Return type differs from its annotation, expected '{expect_return}', but got '{ret}'")
                ret = expect_return
            env[str(str_name)] = lambda: FunctionType([
                new_env[str(param)]() for param in func_params.children
            ], ret)
            return Type("nil")
    if tree.data == "chunk":
        new_env = env.copy()
        for statement in tree.children:
            type = infer(statement, new_env, typeenv)
            if statement.data == "return_stmt":
                return type
        return Type("nil")        
    if tree.data == "return_stmt":
        # TODO: allow multiple return types
        return infer(tree.children[0].children[0], env)
    if tree.data == "array_type":
        return TableType("table", Type("integer"), infer(tree.children[0], env, typeenv))
    if tree.data == "dict_type":
        return TableType("table", infer(tree.children[0], env, typeenv), infer(tree.children[1], env, typeenv))
    if tree.data == "obj_type":
        fields3 = {}
        def run(field):
            nonlocal fields3
            name, type = field.children
            fields3[str(name)] = lambda: infer(type, env, typeenv)
        for field in tree.children:
            run(field)
        return TableType("table", Type("string"), Type("any"), fields3)
    if tree.data == "alias_type":
        name, generics, type = tree.children
        new_typeenv = typeenv.copy()
        names = []
        for g in generics.children:
            if len(g.children) == 1:
                g = g.children[0]
                new_typeenv[str(g)] = lambda: GenericType(str(g))
                names.append(str(g))
                continue
            if g.data == "empty": continue
            @lambda _: _(g)
            def run_2(g):
                g, constraint = g.children
                constraint = infer(constraint, env, new_typeenv)
                new_typeenv[str(g)] = lambda: GenericType(str(g), constraint)
                names.append(str(g))
        new_type = Type("any")
        new_typeenv[str(name)] = lambda: new_type
        new_type = infer(type, env, new_typeenv)
        new_type.set_alias(str(name), [new_typeenv[name]() for name in names])
        typeenv[str(name)] = lambda: new_type
        return new_type
    if tree.data == "record_type":
        name, *fields = tree.children
        new_fields = {}
        def run(field):
            key, val = field.children
            new_fields[str(key)] = lambda: infer(val, env, typeenv)
        map(run, fields)
        type = TableType("table", Type("string"), Type("any"), new_fields)
        typeenv[str(name)] = lambda: type
        return type
    if tree.data == "union_type":
        left, right = tree.children
        left = infer(left, env, typeenv)
        right = infer(right, env, typeenv)
        return UnionType("union", left, right)
    if tree.data == "type_hint":
        return infer(tree.children[0], env, typeenv)
    if tree.data == "generic_type":
        name, *args = tree.children
        name_loc = loc(name)
        args = [infer(a, env, typeenv) for a in args]
        type = typeenv[str(name)]()
        assert len(args) == len(type.generics)
        subs = {}
        i = 0
        for g, arg in zip(type.generics, args):
            assert isinstance(g, GenericType)
            if not extends(arg, g.gconstraint):
                raise ValueError(f"{name_loc} Generic argument '{arg}' does not follow its generic constraint '{g}'")
            subs[str(g.gname)] = arg
            i += 1
        type = apply(type, subs)
        type.args = args
        return type
    if tree.data == "meta_type":
        base, mt = tree.children
        base = infer(base, env, typeenv)
        mt = infer(mt, env, typeenv)
        if not isinstance(base, TableType) or not isinstance(mt, TableType):
            raise ValueError(f"{loc(tree)} Meta type must be a table")
        return MetaType(base, mt)
    if tree.data == "func_type":
        *params, returns = tree.children
        params = [infer(p, env, typeenv) for p in params]
        returns = infer(returns, env, typeenv)
        return FunctionType(params, returns)
    assert False, f"Not implemented: {tree.data}"

with open("test.lua") as f:
    code = f.read()

print(infer(parser.parse(code)))

"""i have a terrible lisp"""

from __future__ import annotations

import ast
import builtins
import importlib.metadata
import operator
import dataclasses
import tokenize
import runpy
import sys
from collections.abc import Callable, Generator, Iterator, Sequence
from functools import reduce
from typing import IO, Any, AnyStr, TypeAlias


__version__ = importlib.metadata.version("lispy")

LispyFunc: TypeAlias = (
    "Callable[[Context, Sequence[Node | Name | Const]], Generator]"
)
stdlib: dict[str, Any] = {}
_Sentinel = object()


@dataclasses.dataclass(slots=True)
class Node:
    """Generic node for S-expressions."""
    children: list[Node | Name | Const]
    parent: Node | None

    def __repr__(self) -> str:
        body = str()
        for i, child in enumerate(self.children):
            if i == 0:
                body += repr(child).strip("@%")
            else:
                body += repr(child)
            body += " "

        return "(" + body.strip() + ")"


@dataclasses.dataclass(slots=True)
class Name:
    """Generic symbol or identifier."""
    value: str

    def __repr__(self) -> str:
        return f"@{self.value}"


@dataclasses.dataclass(slots=True)
class Const:
    """Generic constant value."""
    value: str | int | float

    def __repr__(self) -> str:
        return f"%{self.value!r}"


@dataclasses.dataclass(slots=True)
class Context:
    """Context object used in execution."""
    scopes: list[dict[str, Any]] = dataclasses.field(
        default_factory=lambda: [{}, stdlib]
    )
    callees: list[LispyFunc] = dataclasses.field(
        default_factory=lambda: []
    )

    def __repr__(self) -> str:
        scopes = str()
        for i in self.scopes:
            if i is stdlib:
                scopes += "Global"
            else:
                scopes += repr(i)
            scopes += ", "
        return f"Context(scopes=[{scopes.strip(', ')}])"


def lispy_parse(readline: Callable[[], bytes]) -> Node:
    root = Node([], None)
    node = root

    # problem?
    for token in tokenize.tokenize(readline):
        type_, string, *_ = token
        if type_ == tokenize.OP and string == "(":
            new = Node([], node)
            node.children.append(new)
            node = new
        if type_ == tokenize.OP and string == ")":
            assert node.parent
            node = node.parent
        if type_ == tokenize.NAME:
            node.children.append(Name(string))
        if type_ in {tokenize.NUMBER, tokenize.STRING}:
            node.children.append(Const(ast.literal_eval(string)))

    return root


def lispy_exec(root: Node, ctx: Context | None = None) -> Iterator[Any]:
    """Execute lispy code."""

    ctx = ctx or Context()

    for node in root.children:
        yield exec_in_trampoline(node, ctx)


def exec_in_trampoline(node: Node | Name | Const, ctx: Context) -> Any:
    stack = [exec_node(node, ctx)]
    return_value: Any = _Sentinel
    while stack:
        try:
            if return_value is _Sentinel:
                stack.append(next(stack[-1]))
            else:
                stack.append(stack[-1].send(return_value))
                return_value = _Sentinel
        except StopIteration as exc:
            stack.pop()
            return_value = exc.value
    return return_value


def exec_node(node: Node | Name | Const, ctx: Context) -> Generator:
    """Main execution function. No longer awfully recursive."""
    if isinstance(node, Node):
        func: LispyFunc = yield exec_node(node.children[0], ctx)
        return (yield func(ctx, node.children[1:]))
    if isinstance(node, Name):
        name = node.value
        for scope in ctx.scopes:
            if name in scope:  # i swear this is faster
                return scope[name]
        raise NameError(f"name {node.value} is not defined")
    if isinstance(node, Const):
        return node.value
    raise TypeError


def std_fn(name: str) -> Callable[[LispyFunc], Any]:
    """Add a decorated function to `stdlib`."""
    def decorator(func: LispyFunc) -> LispyFunc:
        stdlib[name] = func
        return func
    return decorator


def eager_fn(func: Callable[..., Any]) -> LispyFunc:
    """Wrapper for functions requiring immediately executed args."""
    def wrapper(ctx: Context, args: Sequence[Any]):
        new_args = []
        for a in args:
            new_args.append((yield exec_node(a, ctx)))
        return func(*new_args)

    return wrapper


def variadic_eager_fn(func: Callable[[Any, Any], Any]) -> LispyFunc:
    def wrapper(ctx: Context, args: Sequence[Any]):
        new_args = []
        for a in args:
            new_args.append((yield exec_node(a, ctx)))
        return reduce(func, new_args)

    return wrapper


# Singletons
stdlib["True"] = True
stdlib["False"] = False
stdlib["None"] = None
stdlib["NotImplemented"] = NotImplemented

# Types
stdlib["Int"] = eager_fn(int)
stdlib["Float"] = eager_fn(float)
stdlib["Complex"] = eager_fn(complex)
stdlib["List"] = eager_fn(lambda *a: list(a))
stdlib["Tuple"] = eager_fn(lambda *a: tuple(a))
stdlib["Set"] = eager_fn(lambda *a: set(a))
stdlib["FrozenSet"] = eager_fn(lambda *a: frozenset(a))

stdlib["Print"] = eager_fn(print)
stdlib["Input"] = eager_fn(input)
stdlib["Abs"] = eager_fn(abs)
stdlib["Max"] = eager_fn(max)
stdlib["Min"] = eager_fn(min)

# Operators
stdlib["Add"] = variadic_eager_fn(operator.add)
stdlib["Sub"] = variadic_eager_fn(operator.sub)
stdlib["Mul"] = variadic_eager_fn(operator.mul)
stdlib["Div"] = variadic_eager_fn(operator.truediv)
stdlib["Mod"] = variadic_eager_fn(operator.mod)
stdlib["Pow"] = eager_fn(operator.pow)
stdlib["Neg"] = eager_fn(operator.neg)
stdlib["AndB"] = variadic_eager_fn(operator.and_)
stdlib["XorB"] = variadic_eager_fn(operator.xor)
stdlib["OrB"] = variadic_eager_fn(operator.or_)
stdlib["InvB"] = eager_fn(operator.inv)
stdlib["LShift"] = eager_fn(operator.lshift)
stdlib["RShift"] = eager_fn(operator.rshift)
stdlib["Eq"] = eager_fn(operator.eq)
stdlib["Not"] = eager_fn(operator.not_)
stdlib["Lt"] = eager_fn(operator.lt)
stdlib["Le"] = eager_fn(operator.le)
stdlib["Neq"] = eager_fn(operator.ne)
stdlib["Gt"] = eager_fn(operator.gt)
stdlib["Ge"] = eager_fn(operator.ge)

# Arrays
stdlib["First"] = eager_fn(lambda l: l[0])
stdlib["Second"] = eager_fn(lambda l: l[1])
stdlib["Third"] = eager_fn(lambda l: l[2])
stdlib["Fourth"] = eager_fn(lambda l: l[3])
stdlib["Fifth"] = eager_fn(lambda l: l[4])
stdlib["Sixth"] = eager_fn(lambda l: l[5])
stdlib["Seventh"] = eager_fn(lambda l: l[6])
stdlib["Eighth"] = eager_fn(lambda l: l[7])
stdlib["Ninth"] = eager_fn(lambda l: l[8])
stdlib["Tenth"] = eager_fn(lambda l: l[9])
stdlib["Last"] = eager_fn(lambda l: l[-1])
stdlib["Nth"] = eager_fn(lambda *a: reduce(lambda l, i: l[i], a[-1:] + a[:-1]))
stdlib["Length"] = eager_fn(len)


@std_fn("Def")
def define(ctx: Context, args: Sequence[Any]):
    """Define a function or variable."""
    if len(args) not in {2, 3} or not isinstance(args[0], Name):
        raise TypeError
    name = args[0].value
    if len(args) == 2:
        if not isinstance(args[1], (Node, Name, Const)):
            raise TypeError
        ctx.scopes[0][name] = (yield exec_node(args[1], ctx))
    elif len(args) == 3:
        if (
            not isinstance(args[1], Node)
            or not isinstance(args[2], (Node, Name, Const))
        ):
            raise TypeError

        func_body = args[2]
        func_defaults: dict[str, Any] = {}
        func_args: list[str] = []

        for i in args[1].children:
            if isinstance(i, Node):
                arg, default = i.children
                if not isinstance(arg, Name):
                    raise TypeError
                func_defaults[arg.value] = yield exec_node(default, ctx)
                func_args.append(arg.value)
            elif isinstance(i, Name):
                func_args.append(i.value)
            else:
                raise TypeError

        def lispy_func(ctx: Context, args: Sequence[Any]):
            func_scope = func_defaults.copy()
            scopes = [func_scope]
            for name, value in zip(func_args, args):
                value = yield exec_node(value, ctx)
                func_scope[name] = value
            func_vars = set(func_scope)
            for scope in ctx.scopes:
                if set(scope) != func_vars:
                    scopes.append(scope)
            callees: list[LispyFunc] = [lispy_func]
            for func in ctx.callees:
                if func is not lispy_func:
                    callees.append(func)
            ctx = Context(scopes, callees)
            return (yield exec_node(func_body, ctx))

        ctx.scopes[0][name] = lispy_func


@std_fn("Let")
def let(ctx: Context, args: Sequence[Any]):
    """Initialize variables in new scope and execute a expression."""
    if len(args) != 2 or not isinstance(args[0], Node):
        raise TypeError

    let_scope: dict[str, Any] = {}
    scopes = [let_scope]
    if isinstance(args[0].children[0], Node):
        declarations = args[0].children
    else:
        declarations = [args[0]]

    for decl in declarations:
        if not isinstance(decl, Node):
            raise TypeError
        *identifiers, value = decl.children
        value = yield exec_node(value, ctx)
        for i in identifiers:
            if not isinstance(i, Name):
                raise TypeError
            let_scope[i.value] = value

    let_vars = set(let_scope)
    for scope in ctx.scopes:
        if set(scope) != let_vars:
            scopes.append(scope)

    return (yield exec_node(args[1], Context(scopes, ctx.callees)))


@std_fn("Lambda")
def lambda_(ctx: Context, args: Sequence[Any]):
    if (
        len(args) != 2
        or not isinstance(args[0], Node)
        or not isinstance(args[1], (Node, Name, Const))
    ):
        raise TypeError

    func_body = args[1]
    func_defaults: dict[str, Any] = {}
    func_args: list[str] = []

    for i in args[0].children:
        if isinstance(i, Node):
            arg, default = i.children
            if not isinstance(arg, Name):
                raise TypeError
            func_defaults[arg.value] = yield exec_node(default, ctx)
            func_args.append(arg.value)
        elif isinstance(i, Name):
            func_args.append(i.value)
        else:
            raise TypeError

    def lispy_func(ctx: Context, args: Sequence[Any]):
        func_scope = func_defaults.copy()
        scopes = [func_scope]
        for name, value in zip(func_args, args):
            value = yield exec_node(value, ctx)
            func_scope[name] = value
        func_vars = set(func_scope)
        for scope in ctx.scopes:
            if set(scope) != func_vars:
                scopes.append(scope)
        callees: list[LispyFunc] = [lispy_func]
        for func in ctx.callees:
            if func is not lispy_func:
                callees.append(func)
        ctx = Context(scopes, callees)
        return (yield exec_node(func_body, ctx))

    return lispy_func


@std_fn("Apply")
def apply(ctx: Context, args: Sequence[Any]):
    if len(args) != 2:
        raise TypeError
    func: LispyFunc = yield exec_node(args[0], ctx)
    func_args: Sequence[Any] = yield exec_node(args[1], ctx)
    return (yield func(ctx, [Const(a) for a in func_args]))


@std_fn("Recall")
def recall(ctx: Context, args: Sequence[Any]):
    return (yield ctx.callees[0](ctx, args))


@std_fn("Include")
def include(ctx: Context, args: Sequence[Any]):
    if len(args) not in {1, 2}:
        raise TypeError

    decl: str = yield exec_node(args[0], ctx)
    if not isinstance(decl, str):
        raise TypeError

    spec = decl.split(".")
    if not (obj := getattr(builtins, spec[0], None)):
        obj = __import__(spec[0])

    for i in spec[1:]:
        obj = getattr(obj, i)

    base = "".join(i.title() for i in spec)
    if len(args) == 2:
        if not isinstance(args[1], Name):
            raise TypeError
        base = args[1].value

    if callable(obj) and not isinstance(obj, type):
        ctx.scopes[0][base] = eager_fn(obj)
        return

    if isinstance(obj, type):
        ctx.scopes[0][base] = eager_fn(obj)
    else:
        ctx.scopes[0][base] = obj

    for attr in dir(obj):
        value = getattr(obj, attr)
        if callable(value):
            value = eager_fn(value)
        ctx.scopes[0][base + attr.title()] = value


@std_fn("If")
def if_(ctx: Context, args: Sequence[Any]) -> Any:
    """Conditionally execute expressions."""
    if len(args) not in {2, 3}:
        raise TypeError
    if (yield exec_node(args[0], ctx)):
        return (yield exec_node(args[1], ctx))
    if len(args) > 2:
        return (yield exec_node(args[2], ctx))
    return None


@std_fn("And")
def and_(ctx: Context, args: Sequence[Any]) -> Any:
    if len(args) != 2:
        raise TypeError
    return (yield exec_node(args[0], ctx)) and (yield exec_node(args[1], ctx))


@std_fn("Or")
def or_(ctx: Context, args: Sequence[Any]) -> Any:
    if len(args) != 2:
        raise TypeError
    return (yield exec_node(args[0], ctx)) or (yield exec_node(args[1], ctx))


@std_fn("List")
@eager_fn
def list_(*args: Any) -> list[Any]:
    """Construct a list from arguments"""
    return list(args)


@std_fn("Tuple")
@eager_fn
def tuple_(*args: Any) -> tuple[Any, ...]:
    """Construct a tuple from arguments"""
    return tuple(args)


@std_fn("Set")
@eager_fn
def set_(*args: Any) -> set[Any]:
    """Construct a set from arguments"""
    return set(args)


@std_fn("ReadLine")
@eager_fn
def read_line(file: IO[AnyStr] | None = None) -> str | bytes:
    return (file or sys.stdin).readline()


frame = sys._getframe(1)
while "importlib" in frame.f_code.co_filename:
    if frame.f_back is None:
        raise RuntimeError
    frame = frame.f_back

filename = frame.f_code.co_filename
if filename == "<stdin>":
    raise RuntimeError(
        "can't work from stdin! "
        "(run `python -m lispy` instead for a REPL)"
    )
if filename != runpy.__file__ and not filename.endswith("lispy"):
    with open(frame.f_code.co_filename, "rb") as src:
        if src.readline() != b"import lispy\n":
            raise SyntaxError("lispy code may only start with import lispy")
        for i in lispy_exec(lispy_parse(src.readline)):
            if i is not None:
                print(i)

    sys.exit()

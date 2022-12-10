"""i have a terrible lisp"""

from __future__ import annotations

import ast
import importlib.metadata
import operator
import dataclasses
import tokenize
import runpy
import sys
from collections import ChainMap
from collections.abc import Callable, Generator, Sequence
from typing import Any, TypeAlias


__version__ = importlib.metadata.version("lispy")

LispyFunc: TypeAlias = "Callable[[Context, Sequence[Any]], Generator]"
stdlib: dict[str, Any] = {}


@dataclasses.dataclass
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


@dataclasses.dataclass
class Name:
    """Generic symbol or identifier."""
    value: str

    def __repr__(self) -> str:
        return f"@{self.value}"


@dataclasses.dataclass
class Const:
    """Generic constant value."""
    value: str | int | float

    def __repr__(self) -> str:
        return f"%{self.value!r}"


@dataclasses.dataclass
class Context:
    """Context object used in execution."""
    scopes: list[dict[str, Any]] = dataclasses.field(
        default_factory=lambda: [stdlib]
    )

    def __repr__(self) -> str:
        scopes = str()
        for i in self.scopes:
            if i is stdlib:
                scopes += "Global"
            else:
                scopes += ", "
        return f"Context(scopes=[{scopes.strip(', ')}])"


def lispy_exec(
    readline: Callable[[], bytes], ctx: Context | None = None
) -> list[Any]:
    """Execute lispy code."""
    root = Node([], None)
    node = root

    for token in tokenize.tokenize(readline):
        t, s, *_ = token

        if t == tokenize.OP and s == "(":
            new = Node([], node)
            node.children.append(new)
            node = new
        if t == tokenize.OP and s == ")":
            assert node.parent
            node = node.parent
        if t == tokenize.NAME:
            node.children.append(Name(s))
        if t in {tokenize.NUMBER, tokenize.STRING}:
            node.children.append(Const(ast.literal_eval(s)))

    res = []
    ctx = ctx or Context()
    for i in root.children:
        res.append(exec_in_trampoline(i, ctx))
    return res


_Sentinel = object()

def exec_in_trampoline(node: Node | Name | Const, ctx: Context) -> Any:
    stack = [exec_node(node, ctx)]
    return_value = _Sentinel
    while stack:
        try:
            if return_value is _Sentinel:
                stack.append(next(stack[-1]))
            else:
                stack.append(stack[-1].send(return_value))
                return_value = _Sentinel
        except StopIteration as e:
            stack.pop()
            return_value = e.value
    return return_value


def exec_node(node: Node | Name | Const, ctx: Context) -> Generator:
    """Main execution function. No longer awfully recursive."""
    if isinstance(node, Node):
        func: LispyFunc = yield exec_node(node.children[0], ctx)
        return (yield func(ctx, node.children[1:]))
    if isinstance(node, Name):
        try:
            return ChainMap(*ctx.scopes)[node.value]
        except KeyError:
            raise NameError(f"name {node.value} is not defined") from None
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


stdlib["True"] = True
stdlib["False"] = False
stdlib["None"] = None
stdlib["Int"] = eager_fn(int)
stdlib["Float"] = eager_fn(float)
stdlib["Print"] = eager_fn(print)
stdlib["Abs"] = eager_fn(abs)
stdlib["Add"] = eager_fn(operator.add)
stdlib["Sub"] = eager_fn(operator.sub)
stdlib["Mul"] = eager_fn(operator.mul)
stdlib["Div"] = eager_fn(operator.truediv)
stdlib["Mod"] = eager_fn(operator.mod)
stdlib["Pow"] = eager_fn(operator.pow)
stdlib["AndB"] = eager_fn(operator.and_)
stdlib["XorB"] = eager_fn(operator.xor)
stdlib["InvB"] = eager_fn(operator.inv)
stdlib["OrB"] = eager_fn(operator.or_)
stdlib["LShift"] = eager_fn(operator.lshift)
stdlib["RShift"] = eager_fn(operator.rshift)
stdlib["Eq"] = eager_fn(operator.eq)
stdlib["Not"] = eager_fn(operator.not_)
stdlib["Lt"] = eager_fn(operator.lt)
stdlib["Le"] = eager_fn(operator.le)
stdlib["Neq"] = eager_fn(operator.ne)
stdlib["Gt"] = eager_fn(operator.gt)
stdlib["Ge"] = eager_fn(operator.ge)


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
            for name, value in zip(func_args, args):
                func_scope[name] = yield exec_node(value, ctx)
            ctx = Context([func_scope, *ctx.scopes])
            return (yield exec_node(func_body, ctx))

        ctx.scopes[0][name] = lispy_func


@std_fn("Let")
def let(ctx: Context, args: Sequence[Any]):
    """Initialize variables in new scope and execute a expression."""
    if len(args) != 2 or not isinstance(args[0], Node):
        raise TypeError

    let_scope = {}
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

    return (yield exec_node(args[1], Context([let_scope, *ctx.scopes])))


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


@std_fn("List")
@eager_fn
def list_(*args: Any) -> list[Any]:
    """Construct a list from arguments"""
    return list(args)


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
        lispy_exec(src.readline)

    sys.exit()

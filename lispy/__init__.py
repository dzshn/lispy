"""i have a terrible lisp"""

from __future__ import annotations

import ast
import operator
import dataclasses
import tokenize
import runpy
import sys
from collections import ChainMap
from collections.abc import Callable, Sequence
from typing import Any, Optional, TypeAlias


LispyFunc: TypeAlias = "Callable[[Context, Sequence[Any]], Any]"
stdlib: dict[str, Any] = {}


@dataclasses.dataclass
class Node:
    """Generic node for S-expressions."""
    children: list[Node | Name | Const]
    parent: Node | None = dataclasses.field(repr=False)


@dataclasses.dataclass
class Name:
    """Generic symbol or identifier."""
    value: str


@dataclasses.dataclass
class Const:
    """Generic constant value."""
    value: str | int | float


@dataclasses.dataclass
class Context:
    """Context object used in execution."""
    scopes: list[dict[str, Any]] = dataclasses.field(
        default_factory=lambda: [stdlib]
    )


def lispy_exec(readline: Callable[[], bytes], ctx: Context | None = None) -> None:
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

    ctx = ctx or Context()
    for i in root.children:
        exec_node(i, ctx)


def exec_node(node: Node | Name | Const, ctx: Context) -> Any:
    """Main execution function. Awfully recursive."""
    if isinstance(node, Node):
        return exec_node(node.children[0], ctx)(ctx, node.children[1:])
    if isinstance(node, Name):
        return ChainMap(*reversed(ctx.scopes))[node.value]
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
        return func(*map(lambda a: exec_node(a, ctx), args))

    return wrapper


stdlib["True"] = True
stdlib["False"] = False
stdlib["None"] = None
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
def define(ctx: Context, args: Sequence[Any]) -> None:
    assert isinstance(args[0], Name)
    name = args[0].value
    if len(args) == 2:
        assert isinstance(args[1], (Node, Name, Const))
        ctx.scopes[-1][name] = exec_node(args[1], ctx)
    elif len(args) == 3:
        assert isinstance(args[1], Node)
        assert isinstance(args[2], (Node, Name, Const))
        func_args: list[str] = []
        for i in args[1].children:
            assert isinstance(i, Name)
            func_args.append(i.value)

        func_body = args[2]

        def lispy_func(ctx: Context, args: Sequence[Any]):
            func_scope = dict(zip(func_args, map(lambda a: exec_node(a, ctx), args)))
            ctx = Context([*ctx.scopes, func_scope])
            return exec_node(func_body, ctx)

        ctx.scopes[-1][name] = lispy_func
    else:
        raise RuntimeError


@std_fn("Let")
def let(ctx: Context, args: Sequence[Any]) -> None:
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
        value = exec_node(value, ctx)
        for i in identifiers:
            if not isinstance(i, Name):
                raise TypeError
            let_scope[i.value] = value

    return exec_node(args[1], Context([*ctx.scopes, let_scope]))


@std_fn("If")
def if_(ctx: Context, args: Sequence[Any]) -> Any:
    if exec_node(args[0], ctx):
        return exec_node(args[1], ctx)
    return exec_node(args[2], ctx)


frame = sys._getframe(1)
while "importlib" in frame.f_code.co_filename:
    if frame.f_back is None:
        raise RuntimeError
    frame = frame.f_back

filename = frame.f_code.co_filename
if filename != runpy.__file__:
    with open(frame.f_code.co_filename, "rb") as src:
        assert src.readline() == b"import lispy\n"
        lispy_exec(src.readline)

    sys.exit()

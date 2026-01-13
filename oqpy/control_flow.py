############################################################################
#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License").
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
############################################################################
"""Context manager objects used for creating control flow contexts."""

from __future__ import annotations

import contextlib
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    Iterator,
    Literal,
    Optional,
    TypeVar,
    overload,
)

from openpulse import ast

from oqpy.base import OQPyExpression, to_ast
from oqpy.classical_types import (
    AstConvertible,
    DurationVar,
    IntVar,
    _ClassicalVar,
    convert_range,
)
from oqpy.timing import convert_float_to_duration

ClassicalVarT = TypeVar("ClassicalVarT", bound=_ClassicalVar)

if TYPE_CHECKING:
    from oqpy.program import Program


__all__ = ["If", "Else", "ForIn", "While", "Range", "Switch", "Case", "Default"]


@contextlib.contextmanager
def If(program: Program, condition: OQPyExpression) -> Iterator[None]:
    """Context manager for doing conditional evaluation.

    .. code-block:: python

        i = IntVar(1)
        with If(program, i == 1):
            program.increment(i)

    """
    program._push()
    yield
    state = program._pop()
    program._state.add_if_clause(to_ast(program, condition), state.body)


@contextlib.contextmanager
def Else(program: Program) -> Iterator[None]:
    """Context manager for conditional evaluation. Must come after an "If" context block.

    .. code-block:: python

        i = IntVar(1)
        with If(program, i == 1):
            program.increment(i, 1)
        with Else(program):
            program.increment(i, 2)

    """
    program._push()
    yield
    state = program._pop()
    program._state.add_else_clause(state.body)


# Overloads needed due mypy bug, see
# github.com/python/mypy/issues/8739
# https://github.com/python/mypy/issues/3737
@overload
def ForIn(
    program: Program,
    iterator: Iterable[AstConvertible] | range | AstConvertible,
    identifier_name: Optional[str],
) -> contextlib._GeneratorContextManager[IntVar]:
    ...  # pragma: no cover


@overload
def ForIn(
    program: Program,
    iterator: Iterable[AstConvertible] | range | AstConvertible,
    identifier_name: Optional[str],
    identifier_type: type[ClassicalVarT],
) -> contextlib._GeneratorContextManager[ClassicalVarT]:
    ...  # pragma: no cover


@contextlib.contextmanager
def ForIn(
    program: Program,
    iterator: Iterable[AstConvertible] | range | AstConvertible,
    identifier_name: Optional[str] = None,
    identifier_type: type[ClassicalVarT] | type[IntVar] = IntVar,
) -> Iterator[ClassicalVarT | IntVar]:
    """Context manager for looping a particular portion of a program.

    .. code-block:: python

        i = IntVar(1)
        with ForIn(program, range(1, 10)) as index:
            program.increment(i, index)

    """
    program._push()
    var = identifier_type(name=identifier_name, needs_declaration=False)
    yield var
    state = program._pop()

    if isinstance(iterator, range):
        # A range can only be iterated over integers.
        assert identifier_type is IntVar, "A range can only be looped over an integer."
        set_declaration = convert_range(program, iterator)
    elif isinstance(iterator, Iterable):
        if identifier_type is DurationVar:
            iterator = (convert_float_to_duration(i) for i in iterator)

        set_declaration = ast.DiscreteSet([to_ast(program, i) for i in iterator])
    else:
        set_declaration = to_ast(program, iterator)

    stmt = ast.ForInLoop(
        identifier_type.type_cls(), var.to_ast(program), set_declaration, state.body
    )
    program._add_statement(stmt)


class Range:
    """AstConvertible which creates an integer range.

    Unlike builtin python range, this allows the components to be AstConvertible,
    instead of just int.
    """

    def __init__(self, start: AstConvertible, stop: AstConvertible, step: AstConvertible = 1):
        self.start = start
        self.stop = stop
        self.step = step

    def to_ast(self, program: Program) -> ast.Expression:
        """Convert to an ast.RangeDefinition."""
        return ast.RangeDefinition(
            to_ast(program, self.start),
            ast.BinaryExpression(
                lhs=to_ast(program, self.stop),
                op=ast.BinaryOperator["-"],
                rhs=ast.IntegerLiteral(value=1),
            ),
            to_ast(program, self.step) if self.step != 1 else None,
        )


@contextlib.contextmanager
def While(program: Program, condition: OQPyExpression) -> Iterator[None]:
    """Context manager for looping a repeating a portion of a program while a condition is True.

    .. code-block:: python

        i = IntVar(1)
        with While(program, i < 5) as index:
            program.increment(i, 1)

    """
    program._push()
    yield
    state = program._pop()
    program._add_statement(ast.WhileLoop(to_ast(program, condition), state.body))


class Switch(contextlib.AbstractContextManager["Switch"]):
    """Context manager for switch statement control flow.

    .. code-block:: python

        selector = IntVar(0)
        with Switch(program, selector) as switch:
            with Case(switch, 0):
                program.increment(result, 1)
            with Case(switch, 1, 2):  # Multiple values in one case
                program.increment(result, 2)
            with Default(switch):
                program.increment(result, 100)

    """

    def __init__(self, program: "Program", target: OQPyExpression):
        self.program = program
        self.target = target
        self.cases: list[tuple[list[ast.Expression], list[ast.Statement]]] = []
        self.default: list[ast.Statement] | None = None

    def __enter__(self) -> "Switch":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> Literal[False]:
        if exc_type is not None:
            return False
        # Build the case tuples as (list of expressions, CompoundStatement)
        case_tuples = [(values, ast.CompoundStatement(body)) for values, body in self.cases]
        default_stmt = ast.CompoundStatement(self.default) if self.default else None
        stmt = ast.SwitchStatement(
            to_ast(self.program, self.target),
            case_tuples,
            default_stmt,
        )
        self.program._add_statement(stmt)
        return False


@contextlib.contextmanager
def Case(switch: Switch, *values: AstConvertible) -> Iterator[None]:
    """Context manager for a case within a switch statement.

    Must be used inside a Switch context. Multiple values can be provided
    for a single case block.

    .. code-block:: python

        with Switch(program, selector) as switch:
            with Case(switch, 0):
                # Handle case 0
                program.increment(result, 1)
            with Case(switch, 1, 2):
                # Handle cases 1 and 2
                program.increment(result, 2)

    """
    if not values:
        raise ValueError("Case requires at least one value")
    switch.program._push()
    yield
    state = switch.program._pop()
    case_values = [to_ast(switch.program, v) for v in values]
    switch.cases.append((case_values, state.body))


@contextlib.contextmanager
def Default(switch: Switch) -> Iterator[None]:
    """Context manager for the default case within a switch statement.

    Must be used inside a Switch context.

    .. code-block:: python

        with Switch(program, selector) as switch:
            with Case(switch, 0):
                program.increment(result, 1)
            with Default(switch):
                # Handle all other cases
                program.increment(result, 100)

    """
    if switch.default is not None:
        raise RuntimeError("Switch statement can only have one default case")
    switch.program._push()
    yield
    state = switch.program._pop()
    switch.default = state.body

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
"""Classes representing variables containing quantum types (i.e. Qubits)."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Iterable, Iterator, Optional, Sequence, Union

from openpulse import ast
from openpulse.printer import dumps

import oqpy
from oqpy.base import (
    AstConvertible,
    HasToAst,
    OQPyExpression,
    Var,
    make_annotations,
    map_to_ast,
    to_ast,
)
from oqpy.classical_types import AngleVar, _ClassicalVar

if TYPE_CHECKING:
    from oqpy.program import Program

__all__ = [
    "Qubit",
    "QubitArray",
    "defcal",
    "gate",
    "PhysicalQubits",
    "Cal",
    "OQPyGateModifier",
    "inv",
    "pow",
    "ctrl",
    "negctrl",
]


class Qubit(Var):
    """OQpy variable representing a single qubit."""

    def __init__(
        self,
        name: str,
        needs_declaration: bool = True,
        annotations: Sequence[str | tuple[str, str]] = (),
    ):
        super().__init__(name, needs_declaration=needs_declaration)
        self.name = name
        self.annotations = annotations

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Qubit) and self.name == other.name

    def __lt__(self, other: Qubit) -> bool:
        return self.name < other.name

    def to_ast(self, prog: Program) -> ast.Expression:
        """Converts the OQpy variable into an ast node."""
        prog._add_var(self)
        return ast.Identifier(self.name)

    def make_declaration_statement(self, program: Program) -> ast.Statement:
        """Make an ast statement that declares the OQpy variable."""
        decl = ast.QubitDeclaration(ast.Identifier(self.name), size=None)
        decl.annotations = make_annotations(self.annotations)
        return decl


class PhysicalQubits:
    """Provides a means of accessing qubit variables corresponding to physical qubits.

    For example, the openqasm qubit "$3" is accessed by ``PhysicalQubits[3]``.
    """

    def __class_getitem__(cls, item: int) -> Qubit:
        assert isinstance(item, int)
        return Qubit(f"${item}", needs_declaration=False)


# Todo (#51): support QubitArray
class QubitArray:
    """Represents an array of qubits."""


class OQPyGateModifier(HasToAst):
    """A generic gate modifier."""

    def __init__(self, modifiers: list[OQPyGateModifier] | None = None) -> None:
        self.modifiers = modifiers if modifiers else [self]
        self.control_qubits: set[Qubit] = (
            self.control_qubits if hasattr(self, "control_qubits") else set()
        )
        self.neg_control_qubits: set[Qubit] = (
            self.neg_control_qubits if hasattr(self, "neg_control_qubits") else set()
        )
        for modifier in self.modifiers:
            neg_ctrl_intersection_set = self.control_qubits.intersection(
                modifier.neg_control_qubits
            )
            ctrl_neg_intersection_set = self.neg_control_qubits.intersection(
                modifier.control_qubits
            )
            overlapping_set = neg_ctrl_intersection_set.union(ctrl_neg_intersection_set)
            if overlapping_set:
                raise ValueError(
                    f"Qubits {[q.name for q in overlapping_set]} can be control and negative"
                    " control qubit at the same time."
                )
            self.control_qubits.update(modifier.control_qubits)
            self.neg_control_qubits.update(modifier.neg_control_qubits)

    def __repr__(self) -> str:
        return " @ ".join([str(modifier) for modifier in self.modifiers])

    def __matmul__(self, rhs: Program | OQPyGateModifier) -> Program | OQPyGateModifier:
        if isinstance(rhs, OQPyGateModifier):
            return OQPyGateModifier(self.modifiers + [rhs])
        elif (
            isinstance(rhs, oqpy.Program)
            and len(rhs._state.body) >= 0
            and isinstance(rhs._state.body[-1], ast.QuantumGate)
        ):
            modifiers_ast = self.to_ast(rhs)
            modifiers_ast = (
                [modifiers_ast]
                if isinstance(modifiers_ast, ast.QuantumGateModifier)
                else modifiers_ast
            )
            rhs._state.body[-1].modifiers = modifiers_ast + rhs._state.body[-1].modifiers
            rhs._state.body[-1].qubits = (
                map_to_ast(rhs, sorted(self.control_qubits))
                + map_to_ast(rhs, sorted(self.neg_control_qubits))
                + rhs._state.body[-1].qubits
            )
            return rhs
        else:
            raise RuntimeError(
                "Gate modifiers cannot be applied to anything else than a gate. Ignoring the modifier."
            )

    def to_ast(self, program: Program) -> ast.Expression:
        """Converts the OQpy object into an ast node."""
        simplified_modifiers: list[OQPyGateModifier] = []
        odd_number_inv = False
        power: AstConvertible = 1.0
        for mod in self.modifiers:
            if isinstance(mod, inv):
                odd_number_inv ^= True
            elif isinstance(mod, pow):
                power = (
                    power * mod.expression  # type: ignore[operator]
                    if not isinstance(power, float) or power != 1.0
                    else mod.expression
                )

        if len(self.control_qubits):
            simplified_modifiers.append(ctrl(self.control_qubits))
        if len(self.neg_control_qubits):
            simplified_modifiers.append(negctrl(self.neg_control_qubits))
        if odd_number_inv:
            simplified_modifiers.append(inv())
        # FIXME: Should we test OQPyExpression or AstConvertible
        if isinstance(power, OQPyExpression) or (isinstance(power, float) and power != 1.0):
            simplified_modifiers.append(pow(power))
        return map_to_ast(program, simplified_modifiers)


class inv(OQPyGateModifier):
    """inv gate modifier."""

    def __repr__(self) -> str:
        return "inv"

    def to_ast(self, program: Program) -> ast.Expression:
        """Converts the OQpy object into an ast node."""
        return ast.QuantumGateModifier(ast.GateModifierName.inv)


class pow(OQPyGateModifier):  # pylint: disable=redefined-builtin
    """pow gate modifier."""

    def __init__(self, expression: AstConvertible) -> None:
        self.expression = expression
        super().__init__()

    def __repr__(self) -> str:
        return f"pow({self.expression})"

    def to_ast(self, program: Program) -> ast.Expression:
        """Converts the OQpy object into an ast node."""
        return ast.QuantumGateModifier(ast.GateModifierName.pow, to_ast(program, self.expression))


class ctrl(OQPyGateModifier):
    """ctrl gate modifier."""

    def __init__(self, qubits: Qubit | Iterable[Qubit]) -> None:
        self.control_qubits = {qubits} if isinstance(qubits, Qubit) else set(qubits)
        super().__init__()

    def __repr__(self) -> str:
        return f"ctrl({', '.join([q.name for q in self.control_qubits])})"

    def to_ast(self, program: Program) -> ast.Expression:
        """Converts the OQpy object into an ast node."""
        return ast.QuantumGateModifier(
            ast.GateModifierName.ctrl,
            to_ast(program, len(self.control_qubits)) if len(self.control_qubits) > 1 else None,
        )


class negctrl(OQPyGateModifier):
    """negctrl gate modifier."""

    def __init__(self, qubits: Qubit | Iterable[Qubit]) -> None:
        self.neg_control_qubits = {qubits} if isinstance(qubits, Qubit) else set(qubits)
        super().__init__()

    def __repr__(self) -> str:
        return f"negctrl({', '.join([q.name for q in self.neg_control_qubits])})"

    def to_ast(self, program: Program) -> ast.Expression:
        """Converts the OQpy object into an ast node."""
        return ast.QuantumGateModifier(
            ast.GateModifierName.negctrl,
            to_ast(program, len(self.neg_control_qubits))
            if len(self.neg_control_qubits) > 1
            else None,
        )


@contextlib.contextmanager
def gate(
    program: Program,
    qubits: Union[Qubit, list[Qubit]],
    name: str,
    arguments: Optional[list[AstConvertible]] = None,
    declare_here: bool = False,
) -> Union[Iterator[None], Iterator[list[AngleVar]], Iterator[AngleVar]]:
    """Context manager for creating a gate.

    .. code-block:: python

        with gate(program, q1, "HRzH", [AngleVar(name="theta")]) as theta:
            program.gate(q1, "H")
            program.gate(q1, "Rz", theta)
            program.gate(q1, "H")
    """
    if isinstance(qubits, Qubit):
        qubits = [qubits]

    arguments_ast = []
    variables = []
    if arguments is not None:
        for arg in arguments:
            if not isinstance(arg, AngleVar):
                raise ValueError(arg, "Gates only support args of type AngleVar.")
            arguments_ast.append(ast.Identifier(name=arg.name))
            arg._needs_declaration = False
            variables.append(arg)

    program._push()
    if len(variables) > 1:
        yield variables
    elif len(variables) == 1:
        yield variables[0]
    else:
        yield None
    state = program._pop()

    stmt = ast.QuantumGateDefinition(
        name=ast.Identifier(name),
        arguments=arguments_ast,
        qubits=[ast.Identifier(q.name) for q in qubits],
        body=state.body,
    )
    if declare_here:
        program._add_statement(stmt)
    program._add_gate(name, stmt, needs_declaration=not declare_here)


@contextlib.contextmanager
def defcal(
    program: Program,
    qubits: Union[Qubit, list[Qubit]],
    name: str,
    arguments: Optional[list[AstConvertible]] = None,
    return_type: Optional[ast.ClassicalType] = None,
) -> Union[Iterator[None], Iterator[list[_ClassicalVar]], Iterator[_ClassicalVar]]:
    """Context manager for creating a defcal.

    .. code-block:: python

        with defcal(program, q1, "X", [AngleVar(name="theta"), oqpy.pi/2], oqpy.bit) as theta:
            program.play(frame, waveform)
    """
    if isinstance(qubits, Qubit):
        qubits = [qubits]
    assert return_type is None or isinstance(return_type, ast.ClassicalType)

    arguments_ast = []
    variables = []
    if arguments is not None:
        for arg in arguments:
            if isinstance(arg, _ClassicalVar):
                arguments_ast.append(
                    ast.ClassicalArgument(type=arg.type, name=ast.Identifier(name=arg.name))
                )
                arg._needs_declaration = False
                variables.append(arg)
            else:
                arguments_ast.append(to_ast(program, arg))

    program._push()
    if len(variables) > 1:
        yield variables
    elif len(variables) == 1:
        yield variables[0]
    else:
        yield None
    state = program._pop()

    stmt = ast.CalibrationDefinition(
        ast.Identifier(name),
        arguments_ast,
        [ast.Identifier(q.name) for q in qubits],
        return_type,
        state.body,
    )
    program._add_statement(stmt)
    program._add_defcal(
        [qubit.name for qubit in qubits], name, [dumps(a) for a in arguments_ast], stmt
    )


@contextlib.contextmanager
def Cal(program: Program) -> Iterator[None]:
    """Context manager that begins a cal block."""
    program._push()
    yield
    state = program._pop()
    program._add_statement(ast.CalibrationStatement(state.body))

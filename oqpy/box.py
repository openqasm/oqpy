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
from typing import TYPE_CHECKING, Iterable, Iterator, Optional, TypeVar, overload

from openpulse import ast

from oqpy.base import OQPyExpression, to_ast
from oqpy.classical_types import (
    AstConvertible,
    DurationVar,
    IntVar,
    _ClassicalVar,
    convert_range,
)
from oqpy.timing import make_duration

ClassicalVarT = TypeVar("ClassicalVarT", bound=_ClassicalVar)

if TYPE_CHECKING:
    from oqpy.program import Program


@contextlib.contextmanager
def Box(program: Program) -> Iterator[None]:
    """Context manager for doing conditional evaluation.

    .. code-block:: python

        with oqpy.Box(program):
            program.gate(oqpy.Qubit("5"), "h")

    """
    program._push()
    yield
    state = program._pop()
    program._add_statement(ast.Box(duration=None, body=state.body))


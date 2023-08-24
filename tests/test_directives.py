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

import copy
import math
import sys
import textwrap
import types
import typing
from dataclasses import dataclass, fields

import numpy as np
import pytest
from openpulse import ast
from openpulse.parser import QASMVisitor
from openpulse.printer import dumps

import oqpy
from oqpy import *
from oqpy.base import OQPyExpression, expr_matches, logical_and, logical_or, OQIndexExpression
from oqpy.quantum_types import PhysicalQubits
from oqpy.timing import OQDurationLiteral


def _type_matches(val, type_hint) -> bool:
    """Return true if the value could be an element of type_hint.

    This is more general than `isinstance` since it handles annotated and union types.
    """
    origin = typing.get_origin(type_hint)
    if origin is None:
        # Make an exception for int where float is requested.
        return isinstance(val, type_hint) or (isinstance(val, int) and issubclass(type_hint, float))
    args = typing.get_args(type_hint)

    union_types = [typing.Union]
    if sys.version_info >= (3, 10):
        union_types.append(types.UnionType)  # 3.9 has no types.UnionType

    if origin in union_types:
        for arg in args:
            if _type_matches(val, arg):
                return True
        return False

    if origin is typing.Annotated:
        return _type_matches(val, args[0])

    if origin is list:
        return isinstance(val, origin) and all(_type_matches(item, args[0]) for item in val)

    if origin is dict:
        return (
            isinstance(val, origin)
            and all(_type_matches(k, args[0]) for k in val.keys())
            and all(_type_matches(v, args[1]) for v in val.values())
        )

    return isinstance(val, origin)


class AstTypeHintChecker(QASMVisitor):
    def __init__(self, except_fields):
        self.except_fields = except_fields

    def generic_visit(self, node, context=None):
        cls = type(node)
        type_hints = typing.get_type_hints(cls)
        for field in fields(cls):
            val = getattr(node, field.name)
            type_hint = type_hints[field.name]
            if not _type_matches(val, type_hint) and field.name not in self.except_fields:
                raise TypeError(
                    f"node of type {type(node).__name__} has type mismatch on field {field.name}\n"
                    f"Got {val} of type {type(val)} but expected something of type {type_hint}"
                )
        super().generic_visit(node, context)


def _check_respects_type_hints(prog, except_fields=()):
    if sys.version_info < (3, 9):
        return  # typing module interface is too different before 3.9
    AstTypeHintChecker(except_fields).visit(prog.to_ast())


def test_version_string():
    prog = Program(version="2.9")

    with pytest.raises(RuntimeError):
        prog = Program("2.x")

    expected = textwrap.dedent(
        """
        OPENQASM 2.9;
        """
    ).strip()

    assert prog.to_qasm() == expected
    _check_respects_type_hints(prog)


def test_variable_declaration():
    b = BoolVar(True, "b")
    i = IntVar(-4, "i")
    u = UintVar(5, "u")
    x = DurationVar(100e-9, "blah")
    y = FloatVar[50](3.3, "y")
    ang = AngleVar(name="ang")
    arr = BitVar[20](name="arr")
    c = BitVar(name="c")
    vars = [b, i, u, x, y, ang, arr, c]

    prog = Program(version=None)
    prog.declare(vars)
    prog.set(arr[1], 0)
    index = IntVar(2, "index")
    prog.set(arr[index], 1)
    prog.set(arr[index + 1], 0)

    with pytest.raises(IndexError):
        prog.set(arr[40], 2)
    with pytest.raises(ValueError):
        BitVar[2.1](name="d")
    with pytest.raises(ValueError):
        BitVar[0](name="d")
    with pytest.raises(ValueError):
        BitVar[-1](name="d")
    with pytest.raises(IndexError):
        prog.set(arr[1.3], 0)
    with pytest.raises(TypeError):
        prog.set(c[0], 1)

    expected = textwrap.dedent(
        """
        int[32] index = 2;
        bool b = true;
        int[32] i = -4;
        uint[32] u = 5;
        duration blah = 100.0ns;
        float[50] y = 3.3;
        angle[32] ang;
        bit[20] arr;
        bit c;
        arr[1] = 0;
        arr[index] = 1;
        arr[index + 1] = 0;
        """
    ).strip()

    assert isinstance(arr[14], OQIndexExpression)
    assert prog.to_qasm() == expected
    _check_respects_type_hints(prog)


def test_complex_numbers_declaration():
    vars = [
        ComplexVar(name="z"),
        ComplexVar(1 + 0j, name="z1"),
        ComplexVar(-1 + 0j, name="z2"),
        ComplexVar(0 + 2j, name="z3"),
        ComplexVar(0 - 2j, name="z4"),
        ComplexVar(1 + 2j, name="z5"),
        ComplexVar(1 - 2j, name="z6"),
        ComplexVar(-1 + 2j, name="z7"),
        ComplexVar(-1 - 2j, name="z8"),
        ComplexVar(1, name="z9"),
        ComplexVar(-1, name="z10"),
        ComplexVar(2j, name="z11"),
        ComplexVar(-2j, name="z12"),
        ComplexVar[float32](1.2 - 2.1j, name="z_with_type1"),
        ComplexVar[float_(16)](1.2 - 2.1j, name="z_with_type2"),
        ComplexVar(1.2 - 2.1j, base_type=float_(16), name="z_with_type3"),
    ]
    with pytest.raises(AssertionError):
        ComplexVar(-2j, base_type=IntVar, name="z12")

    prog = Program(version=None)
    prog.declare(vars)

    expected = textwrap.dedent(
        """
        complex[float[64]] z;
        complex[float[64]] z1 = 1.0;
        complex[float[64]] z2 = -1.0;
        complex[float[64]] z3 = 2.0im;
        complex[float[64]] z4 = -2.0im;
        complex[float[64]] z5 = 1.0 + 2.0im;
        complex[float[64]] z6 = 1.0 - 2.0im;
        complex[float[64]] z7 = -1.0 + 2.0im;
        complex[float[64]] z8 = -1.0 - 2.0im;
        complex[float[64]] z9 = 1.0;
        complex[float[64]] z10 = -1.0;
        complex[float[64]] z11 = 2.0im;
        complex[float[64]] z12 = -2.0im;
        complex[float[32]] z_with_type1 = 1.2 - 2.1im;
        complex[float[16]] z_with_type2 = 1.2 - 2.1im;
        complex[float[16]] z_with_type3 = 1.2 - 2.1im;
        """
    ).strip()

    assert prog.to_qasm() == expected
    _check_respects_type_hints(prog)


def test_array_declaration():
    b = ArrayVar(name="b", init_expression=[True, False], dimensions=[2], base_type=BoolVar)
    i = ArrayVar(name="i", init_expression=[0, 1, 2, 3, 4], dimensions=[5], base_type=IntVar)
    i55 = ArrayVar(
        name="i55", init_expression=[0, 1, 2, 3, 4], dimensions=[5], base_type=IntVar[55]
    )
    u = ArrayVar(name="u", init_expression=[0, 1, 2, 3, 4], dimensions=[5], base_type=UintVar)
    x = ArrayVar(
        name="x", init_expression=[0e-9, 1e-9, 2e-9], dimensions=[3], base_type=DurationVar
    )
    y = ArrayVar(name="y", init_expression=[0.0, 1.0, 2.0, 3.0], dimensions=[4], base_type=FloatVar)
    ang = ArrayVar(
        name="ang", init_expression=[0.0, 1.0, 2.0, 3.0], dimensions=[4], base_type=AngleVar
    )
    comp = ArrayVar(name="comp", init_expression=[0, 1 + 1j], dimensions=[2], base_type=ComplexVar)
    comp55 = ArrayVar(
        name="comp55", init_expression=[0, 1 + 1j], dimensions=[2], base_type=ComplexVar[float_(55)]
    )
    ang_partial = ArrayVar[AngleVar, 2](name="ang_part", init_expression=[oqpy.pi, oqpy.pi / 2])
    simple = ArrayVar[FloatVar](name="no_init", dimensions=[5])
    multidim = ArrayVar[FloatVar[32], 3, 2](
        name="multiDim", init_expression=[[1.1, 1.2], [2.1, 2.2], [3.1, 3.2]]
    )
    npinit = ArrayVar(
        name="npinit",
        init_expression=np.array([0, 1, 2, 4]) * 1e-9,
        dimensions=[11],
        base_type=DurationVar,
    )

    vars = [b, i, i55, u, x, y, ang, comp, comp55, ang_partial, simple, multidim, npinit]

    prog = oqpy.Program(version=None)
    prog.declare(vars)
    prog.set(i[1], 0)  # Set with literal values
    idx = IntVar(name="idx", init_expression=5)
    val = IntVar(name="val", init_expression=10)
    d = DurationVar(name="d", init_expression=0)
    prog.set(i[idx], val)
    prog.set(npinit[5], d - 2e-9)
    prog.set(npinit[0], 2 * npinit[0] + 2e-9)

    expected = textwrap.dedent(
        """
        int[32] idx = 5;
        int[32] val = 10;
        duration d = 0.0ns;
        array[bool, 2] b = {true, false};
        array[int[32], 5] i = {0, 1, 2, 3, 4};
        array[int[55], 5] i55 = {0, 1, 2, 3, 4};
        array[uint[32], 5] u = {0, 1, 2, 3, 4};
        array[duration, 3] x = {0.0ns, 1.0ns, 2.0ns};
        array[float[64], 4] y = {0.0, 1.0, 2.0, 3.0};
        array[angle[32], 4] ang = {0.0, 1.0, 2.0, 3.0};
        array[complex[float[64]], 2] comp = {0, 1.0 + 1.0im};
        array[complex[float[55]], 2] comp55 = {0, 1.0 + 1.0im};
        array[angle[32], 2] ang_part = {pi, pi / 2};
        array[float[64], 5] no_init;
        array[float[32], 3, 2] multiDim = {{1.1, 1.2}, {2.1, 2.2}, {3.1, 3.2}};
        array[duration, 11] npinit = {0.0ns, 1.0ns, 2.0ns, 4.0ns};
        i[1] = 0;
        i[idx] = val;
        npinit[5] = d - 2.0ns;
        npinit[0] = 2 * npinit[0] + 2.0ns;
        """
    ).strip()

    assert prog.to_qasm() == expected
    _check_respects_type_hints(prog)


def test_non_trivial_array_access():
    prog = oqpy.Program()
    port = oqpy.PortVar(name="my_port")
    frame = oqpy.FrameVar(name="my_frame", port=port, frequency=1e9, phase=0)

    zero_to_one = oqpy.ArrayVar(
        name="duration_array",
        init_expression=[0.0, 0.25, 0.5, 0.75, 1],
        dimensions=[5],
        base_type=oqpy.DurationVar,
    )
    one_second = oqpy.DurationVar(init_expression=1, name="one_second")

    one = oqpy.IntVar(name="one", init_expression=1)

    with oqpy.ForIn(prog, range(4), "idx") as idx:
        prog.delay(zero_to_one[idx + one] + one_second, frame)
        prog.set(zero_to_one[idx], 5)

    expected = textwrap.dedent(
        """
        OPENQASM 3.0;
        port my_port;
        array[duration, 5] duration_array = {0.0ns, 250.0ms, 500.0ms, 750.0ms, 1s};
        int[32] one = 1;
        duration one_second = 1s;
        frame my_frame = newframe(my_port, 1000000000.0, 0);
        for int idx in [0:3] {
            delay[duration_array[idx + one] + one_second] my_frame;
            duration_array[idx] = 5;
        }
        """
    ).strip()

    assert prog.to_qasm() == expected
    _check_respects_type_hints(prog)


def test_non_trivial_variable_declaration():
    prog = Program()
    z1 = ComplexVar(5, "z1")
    z2 = ComplexVar(2 * z1, "z2")
    z3 = ComplexVar(z2 + 2j, "z3")
    vars = [z1, z2, z3]
    prog.declare(vars)

    expected = textwrap.dedent(
        """
        OPENQASM 3.0;
        complex[float[64]] z1 = 5.0;
        complex[float[64]] z2 = 2 * z1;
        complex[float[64]] z3 = z2 + 2.0im;
        """
    ).strip()

    assert prog.to_qasm() == expected
    _check_respects_type_hints(prog)


def test_variable_assignment():
    prog = Program()
    i = IntVar(5, name="i")
    prog.set(i, 8)
    prog.set(i.to_ast(prog), 1)
    prog.increment(i, 3)
    prog.mod_equals(i, 2)

    with pytest.raises(TypeError):
        prog.set(i, None)

    expected = textwrap.dedent(
        """
        OPENQASM 3.0;
        int[32] i = 5;
        i = 8;
        i = 1;
        i += 3;
        i %= 2;
        """
    ).strip()

    assert prog.to_qasm() == expected
    _check_respects_type_hints(prog)


def test_binary_expressions():
    prog = Program()
    i = IntVar(5, "i")
    j = IntVar(2, "j")
    k = IntVar(0, "k")
    f = FloatVar(0.0, "f")
    b1 = BoolVar(False, "b1")
    b2 = BoolVar(True, "b2")
    b3 = BoolVar(False, "b3")
    d = DurationVar(5e-9, "d")
    prog.set(i, 2 * (i + j))
    prog.set(j, 2 % (2 - i) % 2)
    prog.set(j, 1 + oqpy.pi)
    prog.set(j, 1 / oqpy.pi**2 / 2 + 2**oqpy.pi)
    prog.set(j, -oqpy.pi * oqpy.pi - i**j)
    prog.set(k, i & 51966)
    prog.set(k, 51966 & i)
    prog.set(k, i & j)
    prog.set(k, i | 51966)
    prog.set(k, 51966 | i)
    prog.set(k, i | j)
    prog.set(k, i ^ 51966)
    prog.set(k, 51966 & i)
    prog.set(k, i ^ j)
    prog.set(k, i >> 1)
    prog.set(k, 1 >> i)
    prog.set(k, i >> j)
    prog.set(k, i << 1)
    prog.set(k, 1 << j)
    prog.set(k, i << j)
    prog.set(k, ~k)
    prog.set(b1, logical_or(b2, b3))
    prog.set(b1, logical_and(b2, True))
    prog.set(b1, logical_or(False, b3))
    prog.set(d, d / 5)
    prog.set(d, d + 5e-9)
    prog.set(d, 5e-9 - d)
    prog.set(d, d + convert_float_to_duration(10e-9))
    prog.set(f, d / convert_float_to_duration(1))

    expected = textwrap.dedent(
        """
        OPENQASM 3.0;
        int[32] i = 5;
        int[32] j = 2;
        int[32] k = 0;
        bool b1 = false;
        bool b2 = true;
        bool b3 = false;
        duration d = 5.0ns;
        float[64] f = 0.0;
        i = 2 * (i + j);
        j = 2 % (2 - i) % 2;
        j = 1 + pi;
        j = 1 / pi ** 2 / 2 + 2 ** pi;
        j = -pi * pi - i ** j;
        k = i & 51966;
        k = 51966 & i;
        k = i & j;
        k = i | 51966;
        k = 51966 | i;
        k = i | j;
        k = i ^ 51966;
        k = 51966 & i;
        k = i ^ j;
        k = i >> 1;
        k = 1 >> i;
        k = i >> j;
        k = i << 1;
        k = 1 << j;
        k = i << j;
        k = ~k;
        b1 = b2 || b3;
        b1 = b2 && true;
        b1 = false || b3;
        d = d / 5;
        d = d + 5.0ns;
        d = 5.0ns - d;
        d = d + 10.0ns;
        f = d / 1s;
        """
    ).strip()

    assert prog.to_qasm() == expected
    _check_respects_type_hints(prog)


@pytest.mark.xfail
def test_add_incomptible_type():
    # This test should fail since we add float to a duration and then don't type cast things
    # properly. This test should be fixed once we land this support.
    prog = oqpy.Program()
    port = oqpy.PortVar(name="my_port")
    frame = oqpy.FrameVar(name="my_frame", port=port, frequency=5e9, phase=0)
    delay = oqpy.DurationVar(10e-9, name="d")
    f = oqpy.FloatVar(5e-9, "f")

    prog.delay(delay + f, frame)

    # Note the automatic conversion of float to duration. Do note that OpenQASM spec does not allows
    # `float * duration` but does allow for `const float * duration`. So this example is not
    # entirely spec-compliant. Though arguably the spec should be changed.
    expected = textwrap.dedent(
        """
        OPENQASM 3.0;
        defcalgrammar "openpulse";
        cal {
            port my_port;
            frame my_frame = newframe(my_port, 5000000000.0, 0);
        }
        duration d = 10.0ns;
        float[64] f = 5e-09;

        delay[d + f * 1s] my_frame;
        """
    ).strip()

    assert prog.to_qasm() == expected
    _check_respects_type_hints(prog)


def test_measure_reset_pragma():
    prog = Program()
    q = PhysicalQubits[0]
    c = BitVar(name="c")
    prog.reset(q)
    prog.pragma("CLASSIFIER linear")
    prog.measure(q, c)
    prog.measure(q)
    with oqpy.If(prog, c == 1):
        with pytest.raises(RuntimeError):
            prog.pragma("Invalid pragma")
        prog.gate(q, "x")
    prog.pragma("LOAD_MEMORY all")

    expected = textwrap.dedent(
        """
        OPENQASM 3.0;
        bit c;
        reset $0;
        pragma CLASSIFIER linear
        c = measure $0;
        measure $0;
        if (c == 1) {
            x $0;
        }
        pragma LOAD_MEMORY all
        """
    ).strip()

    assert prog.to_qasm() == expected
    # Todo: Pragmas aren't currently statements
    _check_respects_type_hints(prog, ["statements"])


def test_bare_if():
    prog = Program()
    i = IntVar(3, "i")
    with If(prog, i <= 0):
        prog.increment(i, 1)
    with If(prog, i != 0):
        prog.set(i, 0)
    with pytest.raises(RuntimeError):
        with If(prog, i < 0 or i == 0):
            prog.increment(i, 1)

    expected = textwrap.dedent(
        """
        OPENQASM 3.0;
        int[32] i = 3;
        if (i <= 0) {
            i += 1;
        }
        if (i != 0) {
            i = 0;
        }
        """
    ).strip()

    assert prog.to_qasm() == expected
    _check_respects_type_hints(prog)


def test_if_else():
    prog = Program()
    i = IntVar(3, "i")
    j = IntVar(2, "j")
    with If(prog, i >= 0):
        with If(prog, j == 0):
            prog.increment(i, 1)
        with Else(prog):
            prog.decrement(i, 1)
    with Else(prog):
        prog.decrement(i, 1)

    with pytest.raises(RuntimeError):
        with Else(prog):
            prog.decrement(i, 1)

    expected = textwrap.dedent(
        """
        OPENQASM 3.0;
        int[32] i = 3;
        int[32] j = 2;
        if (i >= 0) {
            if (j == 0) {
                i += 1;
            } else {
                i -= 1;
            }
        } else {
            i -= 1;
        }
        """
    ).strip()

    assert prog.to_qasm() == expected
    _check_respects_type_hints(prog)


def test_for_in():
    prog = Program()
    j = IntVar(0, "j")
    wf = WaveformVar([0.1, -1.2, 1.3, 2.4], name="wf")
    with ForIn(prog, range(5), "i") as i:
        prog.increment(j, i)
    with ForIn(prog, [-1, 1, -1, 1], "k") as k:
        prog.decrement(j, k)
    with ForIn(prog, np.array([0, 3]), "l") as l:
        prog.set(j, l)
    with ForIn(prog, wf, "m") as m:
        prog.set(j, m)

    expected = textwrap.dedent(
        """
        OPENQASM 3.0;
        int[32] j = 0;
        waveform wf = {0.1, -1.2, 1.3, 2.4};
        for int i in [0:4] {
            j += i;
        }
        for int k in {-1, 1, -1, 1} {
            j -= k;
        }
        for int l in {0, 3} {
            j = l;
        }
        for int m in wf {
            j = m;
        }
        """
    ).strip()

    assert prog.to_qasm() == expected
    _check_respects_type_hints(prog)


def test_for_in_var_types():
    port = oqpy.PortVar("my_port")
    frame = oqpy.FrameVar(port, 3e9, 0, "my_frame")

    # Test over floating point array.
    program = oqpy.Program()
    frequencies = [0.1, 0.2, 0.5]
    with oqpy.ForIn(program, frequencies, "frequency", FloatVar) as f:
        program.set_frequency(frame, f)

    expected = textwrap.dedent(
        """
        OPENQASM 3.0;
        port my_port;
        frame my_frame = newframe(my_port, 3000000000.0, 0);
        for float frequency in {0.1, 0.2, 0.5} {
            set_frequency(my_frame, frequency);
        }
        """
    ).strip()

    assert program.to_qasm() == expected
    _check_respects_type_hints(program)

    # Test over duration array.
    program = oqpy.Program()
    delays = [1e-9, 2e-9, 5e-9, 10e-9, 1e-6]

    with oqpy.ForIn(program, delays, "d", DurationVar) as delay:
        program.delay(delay, frame)

    expected = textwrap.dedent(
        """
        OPENQASM 3.0;
        port my_port;
        frame my_frame = newframe(my_port, 3000000000.0, 0);
        for duration d in {1.0ns, 2.0ns, 5.0ns, 10.0ns, 1.0us} {
            delay[d] my_frame;
        }
        """
    ).strip()

    # Test over angle array
    program = oqpy.Program()
    phases = [0] + [oqpy.pi / i for i in range(10, 1, -2)]

    with oqpy.ForIn(program, phases, "phi", AngleVar) as phase:
        program.set_phase(phase, frame)

    expected = textwrap.dedent(
        """
        OPENQASM 3.0;
        port my_port;
        frame my_frame = newframe(my_port, 3000000000.0, 0);
        for angle phi in {0, pi / 10, pi / 8, pi / 6, pi / 4, pi / 2} {
            set_phase(phi, my_frame);
        }
        """
    ).strip()
    assert program.to_qasm() == expected
    _check_respects_type_hints(program)

    # Test indexing over an ArrayVar
    program = oqpy.Program()
    pyphases = [0] + [oqpy.pi / i for i in range(10, 1, -2)]
    phases = ArrayVar(
        name="phases", dimensions=[len(pyphases)], init_expression=pyphases, base_type=AngleVar
    )

    with oqpy.ForIn(program, range(len(pyphases)), "idx") as idx:
        program.shift_phase(phases[idx], frame)

    expected = textwrap.dedent(
        """
        OPENQASM 3.0;
        port my_port;
        array[angle[32], 6] phases = {0, pi / 10, pi / 8, pi / 6, pi / 4, pi / 2};
        frame my_frame = newframe(my_port, 3000000000.0, 0);
        for int idx in [0:5] {
            shift_phase(phases[idx], my_frame);
        }
        """
    ).strip()

    assert program.to_qasm() == expected
    _check_respects_type_hints(program)


def test_while():
    prog = Program()
    j = IntVar(0, "j")
    with While(prog, j < 5):
        prog.increment(j, 1)
    with While(prog, j > 0):
        prog.decrement(j, 1)

    expected = textwrap.dedent(
        """
        OPENQASM 3.0;
        int[32] j = 0;
        while (j < 5) {
            j += 1;
        }
        while (j > 0) {
            j -= 1;
        }
        """
    ).strip()

    assert prog.to_qasm() == expected
    _check_respects_type_hints(prog)


def test_create_frame():
    prog = Program()
    port = PortVar("storage")
    storage_frame = FrameVar(port, 6e9, name="storage_frame")
    readout_frame = FrameVar(name="readout_frame")
    prog.declare([storage_frame, readout_frame])

    with pytest.raises(ValueError):
        frame = FrameVar(port, name="storage_frame")

    expected = textwrap.dedent(
        """
        OPENQASM 3.0;
        port storage;
        frame storage_frame = newframe(storage, 6000000000.0, 0);
        frame readout_frame;
        """
    ).strip()

    assert prog.to_qasm() == expected
    _check_respects_type_hints(prog)


def test_subroutine_with_return():
    prog = Program()

    @subroutine
    def multiply(prog: Program, x: IntVar, y: IntVar) -> IntVar:
        return x * y

    y = IntVar(2, "y")
    prog.set(y, multiply(prog, y, 3))

    @subroutine
    def declare(prog: Program, x: IntVar):
        prog.declare([x])

    # This won't define a subroutine because it was not called with do_expression.
    # The call is NOT added to the program neither
    declare(prog, y)

    @subroutine
    def delay50ns(prog: Program, q: Qubit) -> None:
        prog.delay(50e-9, q)

    q = PhysicalQubits[0]
    prog.do_expression(delay50ns(prog, q))

    with pytest.raises(ValueError):

        @subroutine
        def return1(prog: Program) -> float:
            return 1.0

        return1(prog)

    with pytest.raises(ValueError):

        @subroutine
        def return2(prog: Program) -> float:
            prog.returns(1.0)

        return2(prog)

    with pytest.raises(ValueError):

        @subroutine
        def add(prog: Program, x: IntVar, y) -> IntVar:
            return x + y

        prog.set(y, add(prog, y, 3))

    expected = textwrap.dedent(
        """
        OPENQASM 3.0;
        def multiply(int[32] x, int[32] y) -> int[32] {
            return x * y;
        }
        def delay50ns(qubit q) {
            delay[50.0ns] q;
        }
        int[32] y = 2;
        y = multiply(y, 3);
        delay50ns($0);
        """
    ).strip()

    assert prog.to_qasm() == expected
    _check_respects_type_hints(prog)


def test_subroutine_order():
    prog = Program()

    @subroutine
    def delay50ns(prog: Program, q: Qubit) -> None:
        prog.delay(50e-9, q)

    @subroutine
    def multiply(prog: Program, x: IntVar, y: IntVar) -> IntVar:
        return x * y

    y = IntVar(2, "y")
    prog.declare([delay50ns, multiply, y])
    prog.set(y, multiply(prog, y, 3))
    q = PhysicalQubits[0]
    prog.do_expression(delay50ns(prog, q))

    expected = textwrap.dedent(
        """
        OPENQASM 3.0;
        def delay50ns(qubit q) {
            delay[50.0ns] q;
        }
        def multiply(int[32] x, int[32] y) -> int[32] {
            return x * y;
        }
        int[32] y = 2;
        y = multiply(y, 3);
        delay50ns($0);
        """
    ).strip()

    assert prog.to_qasm() == expected
    _check_respects_type_hints(prog)


def test_box_and_timings():
    constant = declare_waveform_generator("constant", [("length", duration), ("iq", complex128)])

    port = PortVar("portname")
    frame = FrameVar(port, 1e9, name="framename")
    prog = Program()
    with Box(prog, 500e-9):
        prog.play(frame, constant(100e-9, 0.5))
        prog.delay(200e-7, frame)
        prog.play(frame, constant(100e-9, 0.5))

    with Box(prog):
        prog.play(frame, constant(200e-9, 0.5))

    with pytest.raises(TypeError):
        f = FloatVar(200e-9, "f", needs_declaration=False)
        convert_float_to_duration(f.to_ast(prog))

    expected = textwrap.dedent(
        """
        OPENQASM 3.0;
        extern constant(duration, complex[float[64]]) -> waveform;
        port portname;
        frame framename = newframe(portname, 1000000000.0, 0);
        box[500.0ns] {
            play(framename, constant(100.0ns, 0.5));
            delay[20.0us] framename;
            play(framename, constant(100.0ns, 0.5));
        }
        box {
            play(framename, constant(200.0ns, 0.5));
        }
        """
    ).strip()

    assert prog.to_qasm() == expected
    # Todo: box only currently technically allows QuantumStatements (i.e. gates)
    _check_respects_type_hints(prog, ["body"])


def test_play_capture():
    port = PortVar("portname")
    frame = FrameVar(port, 1e9, name="framename")
    prog = Program()
    constant = declare_waveform_generator("constant", [("length", duration), ("iq", complex128)])

    prog.play(frame, constant(1e-6, 0.5))
    kernel = WaveformVar(constant(1e-6, iq=1), "kernel")
    prog.capture(frame, kernel)

    expected = textwrap.dedent(
        """
        OPENQASM 3.0;
        extern constant(duration, complex[float[64]]) -> waveform;
        port portname;
        frame framename = newframe(portname, 1000000000.0, 0);
        waveform kernel = constant(1.0us, 1);
        play(framename, constant(1.0us, 0.5));
        capture(framename, kernel);
        """
    ).strip()

    assert prog.to_qasm() == expected
    _check_respects_type_hints(prog)


def test_set_shift_frequency():
    port = PortVar("portname")
    frame = FrameVar(port, 1e9, name="framename")
    prog = Program()

    prog.set_frequency(frame, 1.1e9)
    prog.shift_frequency(frame, 0.2e9)

    expected = textwrap.dedent(
        """
        OPENQASM 3.0;
        port portname;
        frame framename = newframe(portname, 1000000000.0, 0);
        set_frequency(framename, 1100000000.0);
        shift_frequency(framename, 200000000.0);
        """
    ).strip()

    assert prog.to_qasm() == expected
    _check_respects_type_hints(prog)


def test_declare_extern():
    program = Program()

    # Test an extern with one input and output
    sqrt = declare_extern("sqrt", [("x", float32)], float32)

    # Test an extern with two inputs and one output
    arctan = declare_extern("arctan", [("x", float32), ("y", float32)], float32)

    # Test an extern with no input and one output
    time = declare_extern("time", [], int32)

    # Test an extern with one input and no output
    set_global_voltage = declare_extern("set_voltage", [("voltage", int32)])

    # Test an extern with no input and no output
    fire_bazooka = declare_extern("fire_bazooka", [])

    f = oqpy.FloatVar(name="f", init_expression=0.0)
    i = oqpy.IntVar(name="i", init_expression=5)

    program.set(f, sqrt(f))
    program.set(f, arctan(f, f))
    program.set(i, time())
    program.do_expression(set_global_voltage(i))
    program.do_expression(fire_bazooka())

    expected = textwrap.dedent(
        """
        OPENQASM 3.0;
        extern sqrt(float[32]) -> float[32];
        extern arctan(float[32], float[32]) -> float[32];
        extern time() -> int[32];
        extern set_voltage(int[32]);
        extern fire_bazooka();
        float[64] f = 0.0;
        int[32] i = 5;
        f = sqrt(f);
        f = arctan(f, f);
        i = time();
        set_voltage(i);
        fire_bazooka();
        """
    ).strip()

    assert program.to_qasm() == expected


def test_defcals():
    prog = Program()
    constant = declare_waveform_generator("constant", [("length", duration), ("iq", complex128)])

    q_port = PortVar("q_port")
    rx_port = PortVar("rx_port")
    tx_port = PortVar("tx_port")
    q_frame = FrameVar(q_port, 6.431e9, name="q_frame")
    rx_frame = FrameVar(rx_port, 5.752e9, name="rx_frame")
    tx_frame = FrameVar(tx_port, 5.752e9, name="tx_frame")

    q1 = PhysicalQubits[1]
    q2 = PhysicalQubits[2]

    with defcal(prog, q2, "x"):
        prog.play(q_frame, constant(1e-6, 0.1))

    with defcal(prog, q2, "rx", [AngleVar(name="theta")]) as theta:
        prog.increment(theta, 0.1)
        prog.play(q_frame, constant(1e-6, 0.1))

    with defcal(prog, q2, "rx", [pi / 3]):
        prog.play(q_frame, constant(1e-6, 0.1))

    with defcal(prog, [q1, q2], "xy", [AngleVar(name="theta"), +pi / 2]) as theta:
        prog.increment(theta, 0.1)
        prog.play(q_frame, constant(1e-6, 0.1))

    with defcal(prog, [q1, q2], "xy", [AngleVar(name="theta"), FloatVar(name="phi"), 10]) as params:
        theta, phi = params
        prog.increment(theta, 0.1)
        prog.increment(phi, 0.2)
        prog.play(q_frame, constant(1e-6, 0.1))

    with defcal(prog, q2, "readout", return_type=oqpy.bit):
        prog.play(tx_frame, constant(2.4e-6, 0.2))
        prog.capture(rx_frame, constant(2.4e-6, 1))

    with pytest.raises(AssertionError):
        with defcal(prog, q2, "readout", return_type=bool):
            prog.play(tx_frame, constant(2.4e-6, 0.2))
            prog.capture(rx_frame, constant(2.4e-6, 1))

    expected = textwrap.dedent(
        """
        OPENQASM 3.0;
        extern constant(duration, complex[float[64]]) -> waveform;
        port rx_port;
        port tx_port;
        port q_port;
        frame q_frame = newframe(q_port, 6431000000.0, 0);
        frame tx_frame = newframe(tx_port, 5752000000.0, 0);
        frame rx_frame = newframe(rx_port, 5752000000.0, 0);
        defcal x $2 {
            play(q_frame, constant(1.0us, 0.1));
        }
        defcal rx(angle[32] theta) $2 {
            theta += 0.1;
            play(q_frame, constant(1.0us, 0.1));
        }
        defcal rx(pi / 3) $2 {
            play(q_frame, constant(1.0us, 0.1));
        }
        defcal xy(angle[32] theta, pi / 2) $1, $2 {
            theta += 0.1;
            play(q_frame, constant(1.0us, 0.1));
        }
        defcal xy(angle[32] theta, float[64] phi, 10) $1, $2 {
            theta += 0.1;
            phi += 0.2;
            play(q_frame, constant(1.0us, 0.1));
        }
        defcal readout $2 -> bit {
            play(tx_frame, constant(2.4us, 0.2));
            capture(rx_frame, constant(2.4us, 1));
        }
        """
    ).strip()
    assert prog.to_qasm() == expected
    _check_respects_type_hints(prog)

    expect_defcal_rx_theta = textwrap.dedent(
        """
        defcal rx(angle[32] theta) $2 {
            theta += 0.1;
            play(q_frame, constant(1.0us, 0.1));
        }
        """
    ).strip()
    assert (
        dumps(prog.defcals[(("$2",), "rx", ("angle[32] theta",))], indent="    ").strip()
        == expect_defcal_rx_theta
    )
    expect_defcal_rx_pio2 = textwrap.dedent(
        """
        defcal rx(pi / 3) $2 {
            play(q_frame, constant(1.0us, 0.1));
        }
        """
    ).strip()
    assert (
        dumps(prog.defcals[(("$2",), "rx", ("pi / 3",))], indent="    ").strip()
        == expect_defcal_rx_pio2
    )
    expect_defcal_xy_theta_pio2 = textwrap.dedent(
        """
        defcal xy(angle[32] theta, pi / 2) $1, $2 {
            theta += 0.1;
            play(q_frame, constant(1.0us, 0.1));
        }
        """
    ).strip()
    assert (
        dumps(
            prog.defcals[(("$1", "$2"), "xy", ("angle[32] theta", "pi / 2"))], indent="    "
        ).strip()
        == expect_defcal_xy_theta_pio2
    )
    expect_defcal_xy_theta_phi = textwrap.dedent(
        """
        defcal xy(angle[32] theta, float[64] phi, 10) $1, $2 {
            theta += 0.1;
            phi += 0.2;
            play(q_frame, constant(1.0us, 0.1));
        }
        """
    ).strip()
    assert (
        dumps(
            prog.defcals[(("$1", "$2"), "xy", ("angle[32] theta", "float[64] phi", "10"))],
            indent="    ",
        ).strip()
        == expect_defcal_xy_theta_phi
    )
    expect_defcal_readout_q2 = textwrap.dedent(
        """
        defcal readout $2 -> bit {
            play(tx_frame, constant(2.4us, 0.2));
            capture(rx_frame, constant(2.4us, 1));
        }
        """
    ).strip()
    assert (
        dumps(prog.defcals[(("$2",), "readout", ())], indent="    ").strip()
        == expect_defcal_readout_q2
    )


def test_returns():
    prog = Program()
    constant = declare_waveform_generator("constant", [("length", duration), ("iq", complex128)])

    rx_port = PortVar("rx_port")
    tx_port = PortVar("tx_port")
    rx_frame = FrameVar(rx_port, 5.752e9, name="rx_frame")
    tx_frame = FrameVar(tx_port, 5.752e9, name="tx_frame")
    capture_v2 = oqpy.declare_extern(
        "capture_v2", [("output", oqpy.frame), ("duration", oqpy.duration)], oqpy.bit
    )

    q0 = PhysicalQubits[0]

    with defcal(prog, q0, "measure_v1", return_type=oqpy.bit):
        prog.play(tx_frame, constant(2.4e-6, 0.2))
        prog.returns(capture_v2(rx_frame, 2.4e-6))

    @subroutine
    def increment_variable_return(prog: Program, i: IntVar) -> IntVar:
        prog.increment(i, 1)
        prog.returns(i)

    j = IntVar(0, name="j")
    k = IntVar(0, name="k")
    prog.declare(j)
    prog.declare(k)
    prog.set(k, increment_variable_return(prog, j))

    expected = textwrap.dedent(
        """
        OPENQASM 3.0;
        extern constant(duration, complex[float[64]]) -> waveform;
        extern capture_v2(frame, duration) -> bit;
        def increment_variable_return(int[32] i) -> int[32] {
            i += 1;
            return i;
        }
        port rx_port;
        port tx_port;
        frame tx_frame = newframe(tx_port, 5752000000.0, 0);
        frame rx_frame = newframe(rx_port, 5752000000.0, 0);
        defcal measure_v1 $0 -> bit {
            play(tx_frame, constant(2.4us, 0.2));
            return capture_v2(rx_frame, 2.4us);
        }
        int[32] j = 0;
        int[32] k = 0;
        k = increment_variable_return(j);
        """
    ).strip()
    assert prog.to_qasm() == expected
    _check_respects_type_hints(prog)

    expected_defcal_measure_v1_q0 = textwrap.dedent(
        """
        defcal measure_v1 $0 -> bit {
            play(tx_frame, constant(2.4us, 0.2));
            return capture_v2(rx_frame, 2.4us);
        }
        """
    ).strip()

    assert (
        dumps(prog.defcals[(("$0",), "measure_v1", ())], indent="    ").strip()
        == expected_defcal_measure_v1_q0
    )

    expected_function_definition = textwrap.dedent(
        """
        def increment_variable_return(int[32] i) -> int[32] {
            i += 1;
            return i;
        }
        """
    ).strip()
    assert (
        dumps(prog.subroutines["increment_variable_return"], indent="    ").strip()
        == expected_function_definition
    )


def test_ramsey_example():
    prog = Program()
    constant = declare_waveform_generator("constant", [("length", duration), ("iq", complex128)])
    gaussian = declare_waveform_generator(
        "gaussian",
        [("length", duration), ("sigma", duration), ("amplitude", float64), ("phase", float64)],
    )
    tx_waveform = constant(2.4e-6, 0.2)

    q_port = PortVar("q_port")
    rx_port = PortVar("rx_port")
    tx_port = PortVar("tx_port")
    ports = [q_port, rx_port, tx_port]

    q_frame = FrameVar(q_port, 6.431e9, name="q_frame")
    rx_frame = FrameVar(rx_port, 5.752e9, name="rx_frame")
    tx_frame = FrameVar(tx_port, 5.752e9, name="tx_frame")
    frames = [q_frame, rx_frame, tx_frame]

    with Cal(prog):
        prog.declare(ports)
        prog.declare(frames)

    q2 = PhysicalQubits[2]

    with defcal(prog, q2, "readout"):
        prog.play(tx_frame, tx_waveform)
        prog.capture(rx_frame, constant(2.4e-6, 1))

    with defcal(prog, q2, "x90"):
        prog.play(q_frame, gaussian(32e-9, 8e-9, 0.2063, 0.0))

    ramsey_delay = DurationVar(12e-6, "ramsey_delay")
    tppi_angle = AngleVar(0, "tppi_angle")

    with Cal(prog):
        with ForIn(prog, range(1001), "shot") as shot:
            prog.declare(ramsey_delay)
            prog.declare(tppi_angle)
            with ForIn(prog, range(81), "delay_increment") as delay_increment:
                (
                    prog.delay(100e-6)
                    .set_phase(q_frame, 0)
                    .set_phase(rx_frame, 0)
                    .set_phase(tx_frame, 0)
                    .gate(q2, "x90")
                    .delay(ramsey_delay)
                    .shift_phase(q_frame, tppi_angle)
                    .gate(q2, "x90")
                    .gate(q2, "readout")
                    .increment(tppi_angle, 20e-9 * 5e6 * 2 * np.pi)
                    .increment(ramsey_delay, 20e-9)
                )

    expected = textwrap.dedent(
        """
        OPENQASM 3.0;
        extern constant(duration, complex[float[64]]) -> waveform;
        extern gaussian(duration, duration, float[64], float[64]) -> waveform;
        cal {
            port q_port;
            port rx_port;
            port tx_port;
            frame q_frame = newframe(q_port, 6431000000.0, 0);
            frame rx_frame = newframe(rx_port, 5752000000.0, 0);
            frame tx_frame = newframe(tx_port, 5752000000.0, 0);
        }
        defcal readout $2 {
            play(tx_frame, constant(2.4us, 0.2));
            capture(rx_frame, constant(2.4us, 1));
        }
        defcal x90 $2 {
            play(q_frame, gaussian(32.0ns, 8.0ns, 0.2063, 0.0));
        }
        cal {
            for int shot in [0:1000] {
                duration ramsey_delay = 12.0us;
                angle[32] tppi_angle = 0;
                for int delay_increment in [0:80] {
                    delay[100.0us];
                    set_phase(q_frame, 0);
                    set_phase(rx_frame, 0);
                    set_phase(tx_frame, 0);
                    x90 $2;
                    delay[ramsey_delay];
                    shift_phase(q_frame, tppi_angle);
                    x90 $2;
                    readout $2;
                    tppi_angle += 0.6283185307179586;
                    ramsey_delay += 20.0ns;
                }
            }
        }
        """
    ).strip()

    expect_defcal_x90_q2 = textwrap.dedent(
        """
        defcal x90 $2 {
            play(q_frame, gaussian(32.0ns, 8.0ns, 0.2063, 0.0));
        }
        """
    ).strip()

    expect_defcal_readout_q2 = textwrap.dedent(
        """
        defcal readout $2 {
            play(tx_frame, constant(2.4us, 0.2));
            capture(rx_frame, constant(2.4us, 1));
        }
        """
    ).strip()

    assert prog.to_qasm() == expected
    _check_respects_type_hints(prog)
    assert dumps(prog.defcals[(("$2",), "x90", ())], indent="    ").strip() == expect_defcal_x90_q2
    assert (
        dumps(prog.defcals[(("$2",), "readout", ())], indent="    ").strip()
        == expect_defcal_readout_q2
    )


def test_rabi_example():
    prog = Program()
    constant = declare_waveform_generator("constant", [("length", duration), ("iq", complex128)])
    gaussian = declare_waveform_generator(
        "gaussian",
        [("length", duration), ("sigma", duration), ("amplitude", float64), ("phase", float64)],
    )

    zcu216_dac231_0 = PortVar("zcu216_dac231_0")
    zcu216_dac230_0 = PortVar("zcu216_dac230_0")
    zcu216_adc225_0 = PortVar("zcu216_adc225_0")
    q0_transmon_xy_frame = FrameVar(zcu216_dac231_0, 3911851971.26885, name="q0_transmon_xy_frame")
    q0_readout_tx_frame = FrameVar(zcu216_dac230_0, 3571600000, name="q0_readout_tx_frame")
    q0_readout_rx_frame = FrameVar(zcu216_adc225_0, 3571600000, name="q0_readout_rx_frame")
    frames = [q0_transmon_xy_frame, q0_readout_tx_frame, q0_readout_rx_frame]
    rabi_pulse_wf = WaveformVar(gaussian(5.2e-8, 1.3e-8, 1.0, 0.0), "rabi_pulse_wf")
    readout_waveform_wf = WaveformVar(constant(1.6e-6, 0.02), "readout_waveform_wf")
    readout_kernel_wf = WaveformVar(constant(1.6e-6, 1), "readout_kernel_wf")
    with ForIn(prog, range(1, 1001), "shot") as shot:
        prog.set_scale(q0_transmon_xy_frame, -0.2)
        with ForIn(prog, range(1, 102), "amplitude") as amplitude:
            prog.delay(200e-6, frames)
            for frame in frames:
                prog.set_phase(frame, 0)
            (
                prog.play(q0_transmon_xy_frame, rabi_pulse_wf)
                .barrier(frames)
                .play(q0_readout_tx_frame, readout_waveform_wf)
                .capture(q0_readout_rx_frame, readout_kernel_wf)
                .barrier(frames)
                .shift_scale(q0_transmon_xy_frame, 0.4 / 100)
            )

    expected = textwrap.dedent(
        """
        OPENQASM 3.0;
        cal {
            port zcu216_adc225_0;
            port zcu216_dac230_0;
            port zcu216_dac231_0;
            frame q0_transmon_xy_frame = newframe(zcu216_dac231_0, 3911851971.26885, 0);
            frame q0_readout_tx_frame = newframe(zcu216_dac230_0, 3571600000, 0);
            frame q0_readout_rx_frame = newframe(zcu216_adc225_0, 3571600000, 0);
            waveform rabi_pulse_wf = gaussian(52.0ns, 13.0ns, 1.0, 0.0);
            waveform readout_waveform_wf = constant(1.6us, 0.02);
            waveform readout_kernel_wf = constant(1.6us, 1);
            for int shot in [1:1000] {
                set_scale(q0_transmon_xy_frame, -0.2);
                for int amplitude in [1:101] {
                    delay[200.0us] q0_transmon_xy_frame, q0_readout_tx_frame, q0_readout_rx_frame;
                    set_phase(q0_transmon_xy_frame, 0);
                    set_phase(q0_readout_tx_frame, 0);
                    set_phase(q0_readout_rx_frame, 0);
                    play(q0_transmon_xy_frame, rabi_pulse_wf);
                    barrier q0_transmon_xy_frame, q0_readout_tx_frame, q0_readout_rx_frame;
                    play(q0_readout_tx_frame, readout_waveform_wf);
                    capture(q0_readout_rx_frame, readout_kernel_wf);
                    barrier q0_transmon_xy_frame, q0_readout_tx_frame, q0_readout_rx_frame;
                    shift_scale(q0_transmon_xy_frame, 0.004);
                }
            }
        }
        """
    ).strip()

    assert prog.to_qasm(encal=True, include_externs=False) == expected
    _check_respects_type_hints(prog)


def test_program_add():
    prog1 = Program()
    constant = declare_waveform_generator("constant", [("length", duration), ("iq", complex128)])

    prog1.delay(1e-6)

    prog2 = Program()
    q1 = PhysicalQubits[1]
    q2 = PhysicalQubits[2]
    port = PortVar("p1")
    frame = FrameVar(port, 5e9, name="f1")
    wf = WaveformVar(constant(100e-9, 0.5), "wf")
    with defcal(prog2, q1, "x180"):
        prog2.play(frame, wf)

    with defcal(prog2, [q1, q2], "two_qubit_gate"):
        prog2.play(frame, wf)
    prog2.gate(q1, "x180")
    i = IntVar(5, "i")
    prog2.declare(i)

    expected = textwrap.dedent(
        """
        OPENQASM 3.0;
        extern constant(duration, complex[float[64]]) -> waveform;
        port p1;
        frame f1 = newframe(p1, 5000000000.0, 0);
        waveform wf = constant(100.0ns, 0.5);
        delay[1.0us];
        defcal x180 $1 {
            play(f1, wf);
        }
        defcal two_qubit_gate $1, $2 {
            play(f1, wf);
        }
        x180 $1;
        int[32] i = 5;
        """
    ).strip()

    prog = prog1 + prog2
    assert prog.to_qasm() == expected
    _check_respects_type_hints(prog)

    with pytest.raises(RuntimeError):
        with If(prog2, i == 0):
            prog = prog1 + prog2

    expected_defcal_two_qubit_gate = textwrap.dedent(
        """
        defcal two_qubit_gate $1, $2 {
            play(f1, wf);
        }
        """
    ).strip()

    assert (
        dumps(prog2.defcals[(("$1", "$2"), "two_qubit_gate", ())], indent="    ").strip()
        == expected_defcal_two_qubit_gate
    )
    assert (
        dumps(prog.defcals[(("$1", "$2"), "two_qubit_gate", ())], indent="    ").strip()
        == expected_defcal_two_qubit_gate
    )


def test_expression_convertible():
    @dataclass
    class A:
        name: str

        def _to_oqpy_expression(self):
            return DurationVar(1e-7, self.name)

    frame = FrameVar(name="f1")
    prog = Program()
    prog.set(A("a1"), 2)
    prog.delay(A("a2"), frame)
    expected = textwrap.dedent(
        """
        OPENQASM 3.0;
        duration a1 = 100.0ns;
        duration a2 = 100.0ns;
        frame f1;
        a1 = 2;
        delay[a2] f1;
        """
    ).strip()
    assert prog.to_qasm() == expected
    _check_respects_type_hints(prog)


def test_waveform_extern_arg_passing():
    prog = Program()
    constant = declare_waveform_generator("constant", [("length", duration), ("iq", complex128)])
    port = PortVar("p1")
    frame = FrameVar(port, 5e9, name="f1")
    prog.play(frame, constant(10e-9, 0.1))
    prog.play(frame, constant(20e-9, iq=0.2))
    prog.play(frame, constant(length=40e-9, iq=0.4))
    prog.play(frame, constant(iq=0.5, length=50e-9))
    with pytest.raises(TypeError):
        prog.play(frame, constant(10e-9, length=10e-9))
    with pytest.raises(TypeError):
        prog.play(frame, constant(10e-9, blah=10e-9))
    with pytest.raises(TypeError):
        prog.play(frame, constant(10e-9, 0.1, 0.1))

    expected = textwrap.dedent(
        """
        OPENQASM 3.0;
        extern constant(duration, complex[float[64]]) -> waveform;
        port p1;
        frame f1 = newframe(p1, 5000000000.0, 0);
        play(f1, constant(10.0ns, 0.1));
        play(f1, constant(20.0ns, 0.2));
        play(f1, constant(40.0ns, 0.4));
        play(f1, constant(50.0ns, 0.5));
        """
    ).strip()

    assert prog.to_qasm() == expected
    _check_respects_type_hints(prog)


def test_needs_declaration():
    prog = Program()
    i1 = IntVar(1, name="i1")
    i2 = IntVar(name="i2", needs_declaration=False)
    p1 = PortVar("p1")
    p2 = PortVar("p2", needs_declaration=False)
    f1 = FrameVar(p1, 5e9, name="f1")
    f2 = FrameVar(p2, 5e9, name="f2", needs_declaration=False)
    q1 = Qubit("q1")
    q2 = Qubit("q2", needs_declaration=False)
    prog.increment(i1, 1)
    prog.increment(i2, 1)
    prog.set_phase(f1, 0)
    prog.set_phase(f2, 0)
    prog.gate(q1, "X")
    prog.gate(q2, "X")

    expected = textwrap.dedent(
        """
        OPENQASM 3.0;
        port p1;
        int[32] i1 = 1;
        frame f1 = newframe(p1, 5000000000.0, 0);
        qubit q1;
        i1 += 1;
        i2 += 1;
        set_phase(f1, 0);
        set_phase(f2, 0);
        X q1;
        X q2;
        """
    ).strip()

    declared_vars = {}
    undeclared_vars = ["i1", "i2", "f1", "f2", "q1", "q2"]
    statement_ast = [
        ast.ClassicalAssignment(
            lvalue=ast.Identifier(name="i1"),
            op=ast.AssignmentOperator["+="],
            rvalue=ast.IntegerLiteral(value=1),
        ),
        ast.ClassicalAssignment(
            lvalue=ast.Identifier(name="i2"),
            op=ast.AssignmentOperator["+="],
            rvalue=ast.IntegerLiteral(value=1),
        ),
        ast.ExpressionStatement(
            expression=ast.FunctionCall(
                name=ast.Identifier(name="set_phase"),
                arguments=[ast.Identifier(name="f1"), ast.IntegerLiteral(value=0)],
            )
        ),
        ast.ExpressionStatement(
            expression=ast.FunctionCall(
                name=ast.Identifier(name="set_phase"),
                arguments=[ast.Identifier(name="f2"), ast.IntegerLiteral(value=0)],
            )
        ),
        ast.QuantumGate(
            modifiers=[],
            name=ast.Identifier(name="X"),
            arguments=[],
            qubits=[ast.Identifier(name="q1")],
            duration=None,
        ),
        ast.QuantumGate(
            modifiers=[],
            name=ast.Identifier(name="X"),
            arguments=[],
            qubits=[ast.Identifier(name="q2")],
            duration=None,
        ),
    ]

    # testing variables before calling to_ast
    assert prog.declared_vars == declared_vars
    assert list(prog.undeclared_vars.keys()) == undeclared_vars
    assert prog._state.body == statement_ast

    assert prog.to_qasm() == expected
    _check_respects_type_hints(prog)

    # testing variables after calling to_ast, checking mutations
    assert prog.declared_vars == declared_vars
    assert list(prog.undeclared_vars.keys()) == undeclared_vars
    assert prog._state.body == statement_ast


def test_discrete_waveform():
    port = PortVar("port")
    frame = FrameVar(port, 5e9, name="frame")
    wfm_float = WaveformVar([-1.2, 1.5, 0.1, 0], name="wfm_float")
    wfm_int = WaveformVar((1, 0, 4, -1), name="wfm_int")
    wfm_complex = WaveformVar(
        np.array([1 + 2j, -1.2j + 3.2, -2.1j, complex(1, 0)]), name="wfm_complex"
    )
    wfm_notype = WaveformVar([0.0, -1j + 0, 1.2 + 0j, -1], name="wfm_notype")

    prog = Program()
    prog.declare([wfm_float, wfm_int, wfm_complex, wfm_notype])
    prog.play(frame, wfm_complex)
    prog.play(frame, [1] * 2 + [0] * 2)

    expected = textwrap.dedent(
        """
        OPENQASM 3.0;
        port port;
        frame frame = newframe(port, 5000000000.0, 0);
        waveform wfm_float = {-1.2, 1.5, 0.1, 0};
        waveform wfm_int = {1, 0, 4, -1};
        waveform wfm_complex = {1.0 + 2.0im, 3.2 - 1.2im, -2.1im, 1.0};
        waveform wfm_notype = {0.0, -1.0im, 1.2, -1};
        play(frame, wfm_complex);
        play(frame, {1, 1, 0, 0});
        """
    ).strip()

    assert prog.to_qasm() == expected
    _check_respects_type_hints(prog)


def test_annotate():
    prog = Program()
    gaussian = declare_waveform_generator(
        "gaussian",
        [("length", duration), ("sigma", duration), ("amplitude", float64), ("phase", float64)],
        annotations=["annotating_extern_decl"],
    )

    some_port = PortVar("some_port", annotations=["makeport", ("some_keyword", "some_command")])
    q0_transmon_xy_frame = FrameVar(
        some_port, 3911851971.26885, name="q0_transmon_xy_frame", annotations=["makeframe"]
    )
    rabi_pulse_wf = WaveformVar(
        gaussian(5.2e-8, 1.3e-8, 1.0, 0.0), "rabi_pulse_wf", annotations=["makepulse"]
    )

    i = IntVar(0, name="i", annotations=["some-int"])
    j = IntVar(0, name="j", annotations=["other-int"])

    q1 = Qubit("q1", annotations=["some_qubit"])
    q2 = Qubit("q2", annotations=["other_qubit"])

    @subroutine(annotations=["inline", ("optimize", "-O3")])
    def f(prog: Program, x: IntVar) -> IntVar:
        return x

    prog.annotate("first-invocation")
    prog.do_expression(f(prog, i))

    prog.annotate("annotation-before-if")
    with If(prog, i != 0):
        prog.annotate("annotation-in-if")
        prog.gate(q1, "x")
    with oqpy.Else(prog):
        prog.annotate(("annotation-in-else"))
        prog.delay(convert_float_to_duration(1e-8), q1)
    prog.annotate("annotation-after-if")

    prog.annotate("annotation-no-else-before-if")
    with If(prog, i != 0):
        prog.annotate("annotation-no-else-in-if")
        prog.gate(q1, "x")
    prog.annotate("annotation-no-else-after-if")

    prog.annotate("make-for-loop", "with additional info")
    with ForIn(prog, range(1, 1001), "shot") as shot:
        prog.annotate("declaring_j")
        prog.declare(j)
        prog.annotate("declaring", "q2")
        prog.declare(q2)
        prog.annotate("make-set-scale")
        prog.set_scale(q0_transmon_xy_frame, -0.2)
        prog.play(q0_transmon_xy_frame, rabi_pulse_wf)
        prog.annotate("playing", "gate")
        prog.gate(q1, "U1")

    prog.annotate("second-invocation")
    prog.set(i, f(prog, i))

    expected = textwrap.dedent(
        """
        OPENQASM 3.0;
        defcalgrammar "openpulse";
        cal {
            @annotating_extern_decl
            extern gaussian(duration, duration, float[64], float[64]) -> waveform;
        }
        @inline
        @optimize -O3
        def f(int[32] x) -> int[32] {
            return x;
        }
        cal {
            @makeport
            @some_keyword some_command
            port some_port;
            @makeframe
            frame q0_transmon_xy_frame = newframe(some_port, 3911851971.26885, 0);
            @makepulse
            waveform rabi_pulse_wf = gaussian(52.0ns, 13.0ns, 1.0, 0.0);
        }
        @some-int
        int[32] i = 0;
        @some_qubit
        qubit q1;
        @first-invocation
        f(i);
        @annotation-before-if
        if (i != 0) {
            @annotation-in-if
            x q1;
        } else {
            @annotation-in-else
            delay[10.0ns] q1;
        }
        @annotation-after-if
        @annotation-no-else-before-if
        if (i != 0) {
            @annotation-no-else-in-if
            x q1;
        }
        @annotation-no-else-after-if
        @make-for-loop with additional info
        for int shot in [1:1000] {
            @declaring_j
            @other-int
            int[32] j = 0;
            @declaring q2
            @other_qubit
            qubit q2;
            @make-set-scale
            set_scale(q0_transmon_xy_frame, -0.2);
            play(q0_transmon_xy_frame, rabi_pulse_wf);
            @playing gate
            U1 q1;
        }
        @second-invocation
        i = f(i);
        """
    ).strip()
    assert prog.to_qasm(encal_declarations=True) == expected
    _check_respects_type_hints(prog)


def test_in_place_subroutine_declaration():
    @subroutine(annotations=["inline", ("optimize", "-O3")])
    def f(prog: Program, x: IntVar) -> IntVar:
        return x

    prog = Program()
    i = IntVar(0, name="i")
    prog.declare([i, f])
    prog.increment(i, 1)

    expected = textwrap.dedent(
        """
        OPENQASM 3.0;
        int[32] i = 0;
        @inline
        @optimize -O3
        def f(int[32] x) -> int[32] {
            return x;
        }
        i += 1;
        """
    ).strip()
    assert prog.to_qasm() == expected
    _check_respects_type_hints(prog)


def test_var_and_expr_matches():
    p1 = PortVar("p1")
    p2 = PortVar("p2")
    f1 = FrameVar(p1, 5e9, name="f1")
    assert f1._var_matches(f1)
    assert f1._var_matches(copy.deepcopy(f1))

    assert expr_matches(f1, f1)
    assert not expr_matches(f1, p1)
    assert not expr_matches(f1, FrameVar(p1, 4e9, name="frame"))
    assert not expr_matches(f1, FrameVar(p2, 5e9, name="frame"))
    assert not expr_matches(BitVar[2]([1, 2], name="a"), BitVar[2]([1], name="a"))

    prog = Program()
    prog.declare(p1)
    assert expr_matches(prog.declared_vars, {"p1": p1})
    assert not expr_matches(prog.declared_vars, {"p2": p1})


def test_program_tracks_frame_waveform_vars():
    prog = Program()

    p1 = PortVar("p1")
    p2 = PortVar("p2")
    p3 = PortVar("p3")
    ports = [p1, p2, p3]

    f1 = FrameVar(p1, 6.431e9, name="f1")
    f2 = FrameVar(p2, 5.752e9, name="f2")

    constant = declare_waveform_generator("constant", [("length", duration), ("iq", complex128)])
    constant_wf = WaveformVar(constant(1.6e-6, 0.02), "constant_wf")

    # No FrameVar or WaveformVar used in the program yet
    assert expr_matches(list(prog.frame_vars), [])
    assert expr_matches(list(prog.waveform_vars), [])

    with Cal(prog):
        prog.declare(ports)
        # add declared vars for FrameVar and WaveformVar
        prog.declare(f1)
        prog.declare(constant_wf)

    q1 = PhysicalQubits[1]

    with defcal(prog, q1, "readout"):
        # use undeclared FrameVar and WaveformVar
        f3 = FrameVar(p3, 5.752e9, name="f3")
        discrete_wf = WaveformVar([-1.2, 1.5, 0.1, 0], name="discrete_wf")
        prog.play(f3, discrete_wf)
        # in-line waveforms will not be tracked by the program
        prog.capture(f2, constant(2.4e-6, 1))

    assert expr_matches(list(prog.frame_vars), [f1, f3, f2])
    assert expr_matches(list(prog.waveform_vars), [constant_wf, discrete_wf])


def test_duration_literal_arithmetic():
    # Test that duration literals can be used as a part of expression.
    port = oqpy.PortVar("myport")
    frame = oqpy.FrameVar(port, 1e9, name="myframe")
    delay_time = oqpy.convert_float_to_duration(50e-9)  # 50 ns
    one_second = oqpy.convert_float_to_duration(1)  # 1 second
    delay_repetition = 10

    program = oqpy.Program()
    repeated_delay = delay_repetition * delay_time
    assert isinstance(repeated_delay, OQPyExpression)
    assert repeated_delay.type == ast.DurationType()

    program.delay(repeated_delay, frame)
    program.shift_phase(frame, 2 * oqpy.pi * (delay_time / one_second))

    expected = textwrap.dedent(
        """
        OPENQASM 3.0;
        port myport;
        frame myframe = newframe(myport, 1000000000.0, 0);
        delay[10 * 50.0ns] myframe;
        shift_phase(myframe, 2 * pi * (50.0ns / 1s));
        """
    ).strip()

    assert program.to_qasm() == expected
    _check_respects_type_hints(program)


def test_make_duration():
    assert expr_matches(convert_float_to_duration(1e-3), OQDurationLiteral(1e-3))
    assert expr_matches(convert_float_to_duration(OQDurationLiteral(1e-4)), OQDurationLiteral(1e-4))

    class MyExprConvertible:
        def _to_oqpy_expression(self):
            return OQDurationLiteral(1e-5)

    assert expr_matches(convert_float_to_duration(MyExprConvertible()), OQDurationLiteral(1e-5))

    class MyToAst:
        def to_ast(self):
            return OQDurationLiteral(1e-6)

    obj = MyToAst()
    assert convert_float_to_duration(obj) is obj

    with pytest.raises(TypeError):
        convert_float_to_duration("asdf")


def test_autoencal():
    port = PortVar("portname")
    frame = FrameVar(port, 1e9, name="framename")
    prog = Program()
    constant = declare_waveform_generator("constant", [("length", duration), ("iq", complex128)])
    i = IntVar(0, "i")

    prog.increment(i, 1)
    with Cal(prog):
        prog.play(frame, constant(1e-6, 0.5))
        kernel = WaveformVar(constant(1e-6, iq=1), "kernel")
        prog.capture(frame, kernel)

    expected = textwrap.dedent(
        """
        OPENQASM 3.0;
        defcalgrammar "openpulse";
        cal {
            extern constant(duration, complex[float[64]]) -> waveform;
            port portname;
            frame framename = newframe(portname, 1000000000.0, 0);
            waveform kernel = constant(1.0us, 1);
        }
        int[32] i = 0;
        i += 1;
        cal {
            play(framename, constant(1.0us, 0.5));
            capture(framename, kernel);
        }
        """
    ).strip()

    assert prog.to_qasm(encal_declarations=True) == expected
    _check_respects_type_hints(prog)


def test_ramsey_example_blog():
    import oqpy

    ramsey_prog = oqpy.Program()  # create a new oqpy program
    qubit = oqpy.PhysicalQubits[1]  # get physical qubit 1
    delay_time = oqpy.DurationVar(0, "delay_time")  # initialize a duration

    # Loop over shots (i.e. repetitions)
    with oqpy.ForIn(ramsey_prog, range(100), "shot_index"):
        ramsey_prog.set(delay_time, 0)  # reset delay time to zero
        # Loop over delays
        with oqpy.ForIn(ramsey_prog, range(101), "delay_index"):
            (
                ramsey_prog.reset(qubit)  # prepare in ground state
                .gate(qubit, "x90")  # pi/2 pulse
                .delay(delay_time, qubit)  # variable delay
                .gate(qubit, "x90")  # pi/2 pulse
                .measure(qubit)  # final measurement
                .increment(delay_time, 100e-9)
            )  # increase delay by 100 ns

    defcals_prog = oqpy.Program()  # create a new oqpy program
    qubit = oqpy.PhysicalQubits[1]  # get physical qubit 1

    # Declare frames: transmon driving frame and readout receive/transmit frames
    xy_frame = oqpy.FrameVar(oqpy.PortVar("dac0"), 6.431e9, name="xy_frame")
    rx_frame = oqpy.FrameVar(oqpy.PortVar("adc0"), 5.752e9, name="rx_frame")
    tx_frame = oqpy.FrameVar(oqpy.PortVar("dac1"), 5.752e9, name="tx_frame")

    # Declare the type of waveform we are working with.
    # It is up to the backend receiving the openqasm to specify
    # what waveforms are allowed. The waveform names and argument types
    # will therefore need to coordinate with the backend.
    constant_waveform = oqpy.declare_waveform_generator(
        "constant",
        [("length", oqpy.duration), ("amplitude", oqpy.float64)],
    )
    gaussian_waveform = oqpy.declare_waveform_generator(
        "gaussian",
        [("length", oqpy.duration), ("sigma", oqpy.duration), ("amplitude", oqpy.float64)],
    )

    with oqpy.defcal(defcals_prog, qubit, "reset"):
        defcals_prog.delay(1e-3)  # reset to ground state by waiting 1 millisecond

    with oqpy.defcal(defcals_prog, qubit, "measure"):
        defcals_prog.play(tx_frame, constant_waveform(2.4e-6, 0.2))
        defcals_prog.capture(rx_frame, constant_waveform(2.4e-6, 1))

    with oqpy.defcal(defcals_prog, qubit, "x90"):
        defcals_prog.play(xy_frame, gaussian_waveform(32e-9, 8e-9, 0.2063))

    full_prog = defcals_prog + ramsey_prog

    expected = textwrap.dedent(
        """
        OPENQASM 3.0;
        defcalgrammar "openpulse";
        cal {
            extern constant(duration, float[64]) -> waveform;
            extern gaussian(duration, duration, float[64]) -> waveform;
            port dac1;
            port adc0;
            port dac0;
            frame tx_frame = newframe(dac1, 5752000000.0, 0);
            frame rx_frame = newframe(adc0, 5752000000.0, 0);
            frame xy_frame = newframe(dac0, 6431000000.0, 0);
        }
        duration delay_time = 0.0ns;
        defcal reset $1 {
            delay[1.0ms];
        }
        defcal measure $1 {
            play(tx_frame, constant(2.4us, 0.2));
            capture(rx_frame, constant(2.4us, 1));
        }
        defcal x90 $1 {
            play(xy_frame, gaussian(32.0ns, 8.0ns, 0.2063));
        }
        for int shot_index in [0:99] {
            delay_time = 0.0ns;
            for int delay_index in [0:100] {
                reset $1;
                x90 $1;
                delay[delay_time] $1;
                x90 $1;
                measure $1;
                delay_time += 100.0ns;
            }
        }
        """
    ).strip()

    assert full_prog.to_qasm(encal_declarations=True) == expected
    _check_respects_type_hints(full_prog)


def test_constant_conversion():
    w = oqpy.FloatVar(math.pi, name="w")
    x = oqpy.FloatVar(3 * math.pi / 4, name="x")
    y = oqpy.FloatVar(math.pi / 2, name="y")
    z = oqpy.FloatVar(7 * math.pi, name="z")
    prog = Program()
    prog.declare([w, x, y, z])
    expected = textwrap.dedent(
        """
        OPENQASM 3.0;
        float[64] w = pi;
        float[64] x = 3 * pi / 4;
        float[64] y = pi / 2;
        float[64] z = 7 * pi;
        """
    ).strip()
    assert prog.to_qasm() == expected
    _check_respects_type_hints(prog)

    prog = Program(simplify_constants=False)
    prog.declare([w, x, y, z])
    expected = textwrap.dedent(
        """
        OPENQASM 3.0;
        float[64] w = 3.141592653589793;
        float[64] x = 2.356194490192345;
        float[64] y = 1.5707963267948966;
        float[64] z = 21.991148575128552;
        """
    ).strip()
    assert prog.to_qasm() == expected
    _check_respects_type_hints(prog)


def test_oqpy_range():
    prog = Program()
    sum = oqpy.IntVar(0, "sum")
    with ForIn(prog, range(10), "i") as i:
        with ForIn(prog, oqpy.Range(1, i), "j") as j:
            prog.increment(sum, j)
    expected = textwrap.dedent(
        """
        OPENQASM 3.0;
        int[32] sum = 0;
        for int i in [0:9] {
            for int j in [1:i - 1] {
                sum += j;
            }
        }
        """
    ).strip()
    assert prog.to_qasm() == expected
    _check_respects_type_hints(prog)


def test_duration_coercion():
    frame = FrameVar(name="f1")
    prog = Program()
    v = oqpy.FloatVar(0.1, name="v")
    prog.delay(v * 100e-9, frame)
    d = oqpy.DurationVar(100e-9, name="d")
    prog.shift_phase(frame, d * 1e4)
    expected = textwrap.dedent(
        """
        OPENQASM 3.0;
        float[64] v = 0.1;
        frame f1;
        duration d = 100.0ns;
        delay[v * 1e-07 * 1s] f1;
        shift_phase(f1, d * 10000.0 / 1s);
        """
    ).strip()
    assert prog.to_qasm() == expected
    _check_respects_type_hints(prog)


def test_io_declaration():
    x = oqpy.DurationVar("input", name="x")
    y = oqpy.FloatVar("output", name="y")
    wf = oqpy.WaveformVar("input", name="wf")
    port = oqpy.PortVar(name="my_port", init_expression="input")
    frame = oqpy.FrameVar(port, 5e9, 0, name="my_frame")

    prog = Program()
    prog.declare(x)
    prog.set(y, 1)
    prog.play(frame, wf)

    expected = textwrap.dedent(
        """
        OPENQASM 3.0;
        input port my_port;
        output float[64] y;
        frame my_frame = newframe(my_port, 5000000000.0, 0);
        input waveform wf;
        input duration x;
        y = 1;
        play(my_frame, wf);
        """
    ).strip()
    assert prog.to_qasm() == expected
    _check_respects_type_hints(prog)


def test_nested_subroutines():
    @oqpy.subroutine
    def f(prog: oqpy.Program) -> oqpy.IntVar:
        i = oqpy.IntVar(name="i", init_expression=1)
        with oqpy.If(prog, i == 1):
            prog.increment(i, 1)
        return i

    @oqpy.subroutine
    def g(prog: oqpy.Program) -> oqpy.IntVar:
        return f(prog)

    prog = oqpy.Program()
    x = oqpy.IntVar(name="x")
    prog.set(x, g(prog))

    expected = textwrap.dedent(
        """
        OPENQASM 3.0;
        def f() -> int[32] {
            int[32] i = 1;
            if (i == 1) {
                i += 1;
            }
            return i;
        }
        def g() -> int[32] {
            return f();
        }
        int[32] x;
        x = g();
        """
    ).strip()

    assert prog.to_qasm() == expected


def test_invalid_gates():
    # missing qubits argument
    prog = oqpy.Program()
    with pytest.raises(TypeError):
        with oqpy.gate(prog, None, "u"):
            pass

    # invalid argument type
    prog = oqpy.Program()
    with pytest.raises(ValueError):
        q = oqpy.Qubit("q", needs_declaration=False)
        with oqpy.gate(prog, q, "u", [oqpy.FloatVar(name="a")]) as a:
            pass


def test_gate_declarations():
    prog = oqpy.Program()
    q = oqpy.Qubit("q", needs_declaration=False)
    with oqpy.gate(
        prog,
        q,
        "u",
        [oqpy.AngleVar(name="alpha"), oqpy.AngleVar(name="beta"), oqpy.AngleVar(name="gamma")],
    ) as (alpha, beta, gamma):
        prog.gate(q, "a", alpha)
        prog.gate(q, "b", beta)
        prog.gate(q, "c", gamma)
        prog.gate(q, "d")
    with oqpy.gate(prog, q, "rz", [oqpy.AngleVar(name="theta")], declare_here=True) as theta:
        prog.gate(q, "u", theta, 0, 0)
    with oqpy.gate(prog, q, "t"):
        prog.gate(q, "rz", oqpy.pi / 4)

    prog.gate(oqpy.PhysicalQubits[1], "t")
    prog.gate(oqpy.PhysicalQubits[2], "t")

    expected = textwrap.dedent(
        """
        OPENQASM 3.0;
        gate u(alpha, beta, gamma) q {
            a(alpha) q;
            b(beta) q;
            c(gamma) q;
            d q;
        }
        gate t q {
            rz(pi / 4) q;
        }
        gate rz(theta) q {
            u(theta, 0, 0) q;
        }
        t $1;
        t $2;
        """
    ).strip()

    assert prog.to_qasm() == expected


def test_qubit_array():
    prog = oqpy.Program()
    q = oqpy.Qubit("q", size=2)
    prog.gate(q[0], "h")
    prog.gate([q[0], q[1]], "cnot")

    expected = textwrap.dedent(
        """
        OPENQASM 3.0;
        qubit[2] q;
        h q[0];
        cnot q[0], q[1];
        """
    ).strip()

    assert prog.to_qasm() == expected

    with pytest.raises(TypeError):
        prog = oqpy.Program()
        q = oqpy.Qubit("q")
        prog.gate(q[0], "h")

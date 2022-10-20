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

from oqpy import (
    FrameVar,
    PortVar,
    Program,
    WaveformVar,
    declare_waveform_generator,
    float64,
)

gaussian_waveform = declare_waveform_generator(
    "gaussian", [("length", float64), ("sigma", float64), ("amplitude", float64)]
)


def create_program(N, inline_waveform):
    prog = Program()
    port = PortVar("port")
    frame = FrameVar(port, 3e9, 0)
    if not inline_waveform:
        waveform = WaveformVar(gaussian_waveform(40e-9, 10e-9, 0.1))

    for i in range(N):
        prog.set_phase(frame, 0)
        prog.set_frequency(frame, 3e9)
        if inline_waveform:
            prog.play(frame, gaussian_waveform(40e-9, 10e-9, 0.1))
        else:
            prog.play(frame, waveform)
    return prog


def serialize_program(N, inline_waveform):
    return create_program(N, inline_waveform).to_qasm()


def test_create_program_inline_waveform():
    create_program(1000, True)


def test_serialize_program_inline_waveform():
    serialize_program(1000, True)


def test_create_program_waveform_var():
    create_program(1000, False)


def test_serialize_program_waveform_var():
    serialize_program(1000, False)


def test_benchmark_create_program_inline_waveform(benchmark):
    benchmark(create_program, 1000, True)


def test_benchmark_serialize_program_inline_waveform(benchmark):
    benchmark(serialize_program, 1000, True)


def test_benchmark_create_program_waveform_var(benchmark):
    benchmark(create_program, 1000, False)


def test_benchmark_serialize_program_waveform_var(benchmark):
    benchmark(serialize_program, 1000, False)

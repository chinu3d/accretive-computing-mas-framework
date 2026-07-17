#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

#include "Agent.hpp"
#include "MicroAgent.hpp"
#include "MacroAgent.hpp"
#include "ContinuousSubstrate.hpp"
#include "LatentMicroAgent.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "Accretive Computing Continuous-Topology Multi-Agent Engine";

    py::class_<Agent, std::shared_ptr<Agent>>(m, "Agent")
        .def("get_wave_function_at", &Agent::get_wave_function_at)
        .def("calculate_overlap_with", &Agent::calculate_overlap_with)
        .def("get_semantic_payload", &Agent::get_semantic_payload)
        .def("set_semantic_payload", &Agent::set_semantic_payload)
        .def("update_semantic_payload", &Agent::update_semantic_payload);

    py::class_<MicroAgent, Agent, std::shared_ptr<MicroAgent>>(m, "MicroAgent")
        .def(py::init<const Eigen::Vector3d&, double, double, double>(),
             py::arg("initial_pos"), py::arg("amplitude"), py::arg("frequency"), py::arg("phase"))
        .def("get_position", &MicroAgent::get_position)
        .def("set_position", &MicroAgent::set_position)
        .def("get_amplitude", &MicroAgent::get_amplitude)
        .def("get_frequency", &MicroAgent::get_frequency)
        .def("get_phase", &MicroAgent::get_phase)
        .def("set_phase", &MicroAgent::set_phase);

    py::class_<LatentMicroAgent, Agent, std::shared_ptr<LatentMicroAgent>>(m, "LatentMicroAgent")
        .def(py::init<const Eigen::VectorXd&, double, double, double>(),
             py::arg("initial_pos"), py::arg("amplitude"), py::arg("frequency"), py::arg("phase"))
        .def("get_position", &LatentMicroAgent::get_position)
        .def("set_position", &LatentMicroAgent::set_position)
        .def("get_amplitude", &LatentMicroAgent::get_amplitude)
        .def("get_frequency", &LatentMicroAgent::get_frequency)
        .def("get_phase", &LatentMicroAgent::get_phase)
        .def("set_phase", &LatentMicroAgent::set_phase);

    py::class_<MacroAgent, Agent, std::shared_ptr<MacroAgent>>(m, "MacroAgent")
        .def(py::init<>())
        .def("cable_bind", &MacroAgent::cable_bind)
        .def("traverse_topology", &MacroAgent::traverse_topology);

    py::class_<ContinuousSubstrate, std::shared_ptr<ContinuousSubstrate>>(m, "ContinuousSubstrate")
        .def(py::init<>())
        .def("inject_agent", &ContinuousSubstrate::inject_agent)
        .def("calculate_free_energy", &ContinuousSubstrate::calculate_free_energy)
        .def("set_mc_resolution", &ContinuousSubstrate::set_mc_resolution, py::arg("samples"))
        .def("get_mc_resolution", &ContinuousSubstrate::get_mc_resolution)
        .def("step_simulation", [](ContinuousSubstrate& self, double dt) {
            // Rule 3: Enforce Python GIL Release
            // This allows the massively parallel threads in step_simulation to run without blocking Python
            py::gil_scoped_release release;
            self.step_simulation(dt);
        }, py::arg("dt"));
}

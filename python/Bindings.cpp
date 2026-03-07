#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "../include/ForwardModel.hpp"

namespace py = pybind11;

PYBIND11_MODULE(rans_sst_py, m) {
    m.doc() = "RANS-SST Bayesian Calibration – C++ Forward Model";
    // EvaluationStatus
    py::enum_<EvaluationStatus>(m, "EvaluationStatus")
        .value("Converged", EvaluationStatus::Converged)
        .value("Unconverged", EvaluationStatus::Unconverged)
        .value("DivergenceDetected", EvaluationStatus::Diverged)
        .value("InvalidParameters", EvaluationStatus::InvalidParameters)
        .value("Unknown", EvaluationStatus::Unknown);

    // EvaluationResult
    py::class_<EvaluationResult>(m, "EvaluationResult")
        .def_readonly("status", &EvaluationResult::status)
        .def_readonly("log_lik", &EvaluationResult::loglik)
        .def_readonly("predictions", &EvaluationResult::predictions)
        .def_readonly("simple_iters", &EvaluationResult::simpleIters);

    // SSTCoefficients
    py::class_<SSTCoefficients>(m, "SSTCoefficients")
        .def(py::init<>())
        .def_readwrite("sigma_k1", &SSTCoefficients::sigma_k1)
        .def_readwrite("sigma_w1", &SSTCoefficients::sigma_w1)
        .def_readwrite("beta1", &SSTCoefficients::beta1)
        .def_readwrite("alpha1", &SSTCoefficients::alpha1)
        .def_readwrite("sigma_k2", &SSTCoefficients::sigma_k2)
        .def_readwrite("sigma_w2", &SSTCoefficients::sigma_w2)
        .def_readwrite("beta2", &SSTCoefficients::beta2)
        .def_readwrite("alpha2", &SSTCoefficients::alpha2)
        .def_readwrite("betaStar", &SSTCoefficients::betaStar)
        .def_readwrite("a1", &SSTCoefficients::a1)
        .def_readwrite("kappa", &SSTCoefficients::kappa);

    // InferenceParameterSet
    py::class_<InferenceParameterSet>(m, "InferenceParameterSet")
        .def(py::init<>())
        .def_readonly("name", &InferenceParameterSet::name)
        .def("n_active", &InferenceParameterSet::nActive)
        .def("active_names", &InferenceParameterSet::activeNames)
        .def("pack", &InferenceParameterSet::pack)
        .def("unpack", &InferenceParameterSet::unpack)
        .def("in_bounds", &InferenceParameterSet::inBounds)
        .def("lower_bounds", &InferenceParameterSet::lowerBounds)
        .def("upper_bounds", &InferenceParameterSet::upperBounds)
        .def_static("a1_betaStar", &InferenceParameterSet::a1_betaStar)
        .def_static("inlet_turb", &InferenceParameterSet::inletTurb)
        .def_static("near_wall4", &InferenceParameterSet::nearWall4)
        .def_static("all11", &InferenceParameterSet::all11);

    // Mesh (minimal exposure)
    py::class_<Mesh>(m, "Mesh")
        .def_static("make_channel_2d", py::overload_cast<int, int, double, double>(&Mesh::makeChannel2D),
             py::arg("nx"), py::arg("ny"), py::arg("Lx"), py::arg("Ly"))
        .def_static("make_channel_2d", py::overload_cast<int, int, double, double, double, double>(&Mesh::makeChannel2D),
             py::arg("nx"), py::arg("ny"), py::arg("Lx"), py::arg("Ly"),
             py::arg("Re"), py::arg("yPlusTarget") = 1.0)
        .def_static("load_from_file", &Mesh::loadFromFile)
        .def("compute_wall_distance", &Mesh::computeWallDistance)
        .def("n_cells", &Mesh::nCells)
        .def("n_faces", &Mesh::nFaces)
        .def("n_patches", &Mesh::nPatches);

    // FlowBoundaryConditions
    py::class_<FlowBoundaryConditions>(m, "FlowBoundaryConditions")
        .def(py::init<>())
        .def_static("channel_defaults", &FlowBoundaryConditions::channelDefaults)
        .def_static("flat_plate_defaults", &FlowBoundaryConditions::flatPlateDefaults);

    // SolverSettings
    py::class_<SolverSettings>(m, "SolverSettings")
        .def(py::init<>())
        .def_readwrite("max_iterations", &SolverSettings::maxIterations)
        .def_readwrite("convergence_tol", &SolverSettings::convergenceTol)
        .def_readwrite("alpha_u", &SolverSettings::alphaU)
        .def_readwrite("alpha_p", &SolverSettings::alphaP)
        .def_readwrite("verbose", &SolverSettings::verbose)
        .def_readwrite("report_interval", &SolverSettings::reportInterval);

    // ObservationOperator
    py::class_<ObservationOperator>(m, "ObservationOperator")
        .def(py::init<>())
        .def("add_drag", &ObservationOperator::addDrag,
             py::arg("wall_patch"), py::arg("cd_obs"), py::arg("sigma"),
             py::arg("ref_area"), py::arg("ref_vel"), py::arg("sigma_model") = 0.0)
        .def("add_skin_friction", &ObservationOperator::addSkinFriction,
             py::arg("wall_patch"), py::arg("location"), py::arg("cf_obs"),
             py::arg("sigma"), py::arg("ref_vel"), py::arg("sigma_model") = 0.0)
        .def("add_velocity_profile", &ObservationOperator::addVelocityProfile,
             py::arg("location"), py::arg("component"), py::arg("u_obs"),
             py::arg("sigma"), py::arg("sigma_model") = 0.0)
        .def("n_obs", &ObservationOperator::nObs);

    // Vec3 (for location arguments)
    py::class_<Vec3>(m, "Vec3")
        .def(py::init<double, double, double>())
        .def_readwrite("x", &Vec3::x)
        .def_readwrite("y", &Vec3::y)
        .def_readwrite("z", &Vec3::z);

    // ForwardModel
    py::class_<ForwardModel>(m, "ForwardModel")
        .def(py::init<const Mesh&, const InferenceParameterSet&,
                       const ObservationOperator&, const FlowBoundaryConditions&,
                       double, const SolverSettings&, const Vec3&,
                       double, double, double>(),
             py::arg("mesh"), py::arg("param_set"), py::arg("obs_op"),
             py::arg("bcs"), py::arg("nu"),
             py::arg("settings") = SolverSettings{},
             py::arg("u_init") = Vec3{1,0,0},
             py::arg("p_init") = 0.0,
             py::arg("k_init") = 1e-4,
             py::arg("omega_init") = 1.0)
        .def("evaluate", &ForwardModel::evaluate)
        .def("penalized_log_likelihood", &ForwardModel::penalizedLogLikelihood)
        .def("param_set", &ForwardModel::paramSet, py::return_value_policy::reference);
}
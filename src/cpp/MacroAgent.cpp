#include "MacroAgent.hpp"
#include "MicroAgent.hpp"
#include <iostream>
#include <cmath>
#include <random>

std::complex<double> MacroAgent::get_wave_function_at(double x, double y, double z, double t) const {
    // A MacroAgent's wave function is the superposition of its constituents
    std::complex<double> total_wave(0.0, 0.0);
    for (const auto& agent : constituents) {
        if (agent) {
            total_wave += agent->get_wave_function_at(x, y, z, t);
        }
    }
    return total_wave;
}

void MacroAgent::cable_bind(std::shared_ptr<Agent> A, std::shared_ptr<Agent> B) {
    // In a full implementation, we might check if they export residual mismatch here,
    // but for now, we just bind them into this nested MacroAgent.
    constituents.push_back(A);
    constituents.push_back(B);
}

void MacroAgent::traverse_topology() const {
    // Recursively process the topological braid
    for (const auto& agent : constituents) {
        if (auto macro_agent = std::dynamic_pointer_cast<MacroAgent>(agent)) {
            macro_agent->traverse_topology();
        } else if (auto micro_agent = std::dynamic_pointer_cast<MicroAgent>(agent)) {
            // Reached base topology (e.g. logging or accumulating data)
             std::cout << "Traversing topology level..." << std::endl;
        }
    }
}

double MacroAgent::calculate_overlap_with(const Agent& other) const {
    double T = 1.0;
    int num_samples = 1000;
    const double space_limit = 10.0;
    double V_total = std::pow(2.0 * space_limit, 3.0) * T;
    
    std::random_device rd;
    std::mt19937 gen(42); 
    std::uniform_real_distribution<> space_dist(-space_limit, space_limit);
    std::uniform_real_distribution<> time_dist(0.0, T);

    std::complex<double> overlap_sum(0.0, 0.0);

    for (int i = 0; i < num_samples; ++i) {
        double x = space_dist(gen);
        double y = space_dist(gen);
        double z = space_dist(gen);
        double t = time_dist(gen);

        std::complex<double> psi_a = this->get_wave_function_at(x, y, z, t);
        std::complex<double> psi_b = other.get_wave_function_at(x, y, z, t);

        overlap_sum += psi_a * std::conj(psi_b);
    }

    std::complex<double> integral_approx = (V_total / num_samples) * overlap_sum;
    return 1.0 - (integral_approx.real() / T);
}

Eigen::VectorXd MacroAgent::get_spatial_features() const {
    if (constituents.empty()) return Eigen::VectorXd();
    
    Eigen::VectorXd sum = constituents[0]->get_spatial_features();
    for (size_t i = 1; i < constituents.size(); ++i) {
        sum += constituents[i]->get_spatial_features();
    }
    return sum / constituents.size();
}

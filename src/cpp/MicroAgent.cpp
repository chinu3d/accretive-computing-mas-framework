#include "MicroAgent.hpp"
#include <cmath>
#include <random>

MicroAgent::MicroAgent(const Eigen::Vector3d& initial_pos, double amplitude, double frequency, double phase)
    : position(initial_pos), A(amplitude), omega(frequency), theta(phase) {}

std::complex<double> MicroAgent::get_wave_function_at(double x, double y, double z, double t) const {
    // Ambient Isotopy (Geometric Oscillation)
    // Core position oscillates: pos_t = position + A * sin(omega * t + theta)
    Eigen::Vector3d oscillation = Eigen::Vector3d::Ones() * (A * std::sin(omega * t + theta));
    Eigen::Vector3d current_center = position + oscillation;

    Eigen::Vector3d eval_point(x, y, z);
    double dist_sq = (eval_point - current_center).squaredNorm();

    // Gaussian wave packet centered at current_center
    double amplitude_envelope = std::exp(-dist_sq);

    // Complex phase component
    std::complex<double> phase_component = std::polar(1.0, omega * t + theta);

    return amplitude_envelope * phase_component;
}

#include <random>

double MicroAgent::calculate_overlap_with(const Agent& other) const {
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

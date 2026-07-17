#include "LatentMicroAgent.hpp"
#include <cmath>

LatentMicroAgent::LatentMicroAgent(const Eigen::VectorXd& initial_pos, double amplitude, double frequency, double phase)
    : position(initial_pos), A(amplitude), omega(frequency), theta(phase) {}

std::complex<double> LatentMicroAgent::get_wave_function_at(double x, double y, double z, double t) const {
    throw std::runtime_error("LatentMicroAgent (N-dimensional) does not support legacy 3D get_wave_function_at()");
}

double LatentMicroAgent::calculate_overlap_with(const Agent& other) const {
    const LatentMicroAgent* latent_other = dynamic_cast<const LatentMicroAgent*>(&other);
    
    if (!latent_other) {
        // If comparing to a 3D legacy agent, overlap is undefined (0)
        return 0.0;
    }
    
    // Gaussian RBF Kernel for analytical N-dimensional overlap
    // overlap = A_a * A_b * exp(-gamma * ||pos_a - pos_b||^2) * cos(theta_a - theta_b)
    
    double dist_sq = (this->position - latent_other->position).squaredNorm();
    double rbf_kernel = std::exp(-0.5 * dist_sq); // gamma = 0.5
    
    double phase_alignment = std::cos(this->theta - latent_other->theta);
    
    double overlap = this->A * latent_other->A * rbf_kernel * phase_alignment;
    
    return overlap;
}

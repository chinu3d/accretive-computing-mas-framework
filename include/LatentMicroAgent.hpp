#pragma once

#include "Agent.hpp"
#include <Eigen/Dense>
#include <stdexcept>

class LatentMicroAgent : public Agent {
public:
    // Allow dynamic size or fixed size VectorXd
    LatentMicroAgent(const Eigen::VectorXd& initial_pos, 
                     double amplitude, 
                     double frequency, 
                     double phase);

    virtual ~LatentMicroAgent() = default;

    // The legacy 3D wave function is invalid in N-D space
    std::complex<double> get_wave_function_at(double x, double y, double z, double t) const override;

    // The analytical overlap in N-dimensional space
    double calculate_overlap_with(const Agent& other) const override;
    
    Eigen::VectorXd get_spatial_features() const override { return position; }

    // Accessors for physics properties
    Eigen::VectorXd get_position() const { return position; }
    void set_position(const Eigen::VectorXd& pos) { position = pos; }

    double get_amplitude() const { return A; }
    double get_frequency() const { return omega; }
    double get_phase() const { return theta; }
    void set_phase(double p) { theta = p; }

private:
    Eigen::VectorXd position; // N-dimensional embedding
    double A;                 // Oscillation amplitude
    double omega;             // Frequency
    double theta;             // Phase
};

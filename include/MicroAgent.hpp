#pragma once

#include "Agent.hpp"
#include <Eigen/Dense>

class MicroAgent : public Agent {
public:
    MicroAgent(const Eigen::Vector3d& initial_pos, 
               double amplitude, 
               double frequency, 
               double phase);

    virtual ~MicroAgent() = default;

    std::complex<double> get_wave_function_at(double x, double y, double z, double t) const override;
    
    double calculate_overlap_with(const Agent& other) const override;
    Eigen::VectorXd get_spatial_features() const override { return position; }

    // Accessors for physics properties
    Eigen::Vector3d get_position() const { return position; }
    void set_position(const Eigen::Vector3d& pos) { position = pos; }

    double get_amplitude() const { return A; }
    double get_frequency() const { return omega; }
    double get_phase() const { return theta; }
    void set_phase(double p) { theta = p; }

private:
    Eigen::Vector3d position; // x0
    double A;                 // Oscillation amplitude
    double omega;             // Frequency
    double theta;             // Phase
};

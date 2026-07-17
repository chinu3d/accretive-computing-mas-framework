#pragma once

#include "Agent.hpp"
#include <vector>
#include <memory>
#include <Eigen/Dense>

class MacroAgent : public Agent {
public:
    MacroAgent() = default;
    virtual ~MacroAgent() = default;

    std::complex<double> get_wave_function_at(double x, double y, double z, double t) const override;
    double calculate_overlap_with(const Agent& other) const override;
    
    // For a macro agent, the center of mass (mean feature vector)
    Eigen::VectorXd get_spatial_features() const override;

    // Recursively binds two agents together
    void cable_bind(std::shared_ptr<Agent> A, std::shared_ptr<Agent> B);

    // Recursively process or retrieve constituent agents
    void traverse_topology() const;

    const std::vector<std::shared_ptr<Agent>>& get_constituents() const { return constituents; }

private:
    std::vector<std::shared_ptr<Agent>> constituents;
};

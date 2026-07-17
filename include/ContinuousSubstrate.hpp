#pragma once

#include "Agent.hpp"
#include <vector>
#include <memory>

class ContinuousSubstrate {
public:
    ContinuousSubstrate();
    ~ContinuousSubstrate() = default;

    void inject_agent(std::shared_ptr<Agent> agent);

    // Advances the continuous simulation using gradient descent
    void step_simulation(double dt);

    // Computes the system's global free energy functional
    double calculate_free_energy() const;

    void set_mc_resolution(int samples) {} // Deprecated
    int get_mc_resolution() const { return 0; }

private:
    std::vector<std::shared_ptr<Agent>> field_agents;
    double stability_threshold = 0.1; // tau
    double dissipation_delta = 0.05;  // Delta
    int mc_resolution = 1000;         // Tunable MC resolution parameter
};

#pragma once

#include <complex>
#include <memory>
#include <string>
#include <unordered_map>
#include <Eigen/Dense>

class Agent {
public:
    virtual ~Agent() = default;

    // The wave function represents the field amplitude at (x, y, z) and time t
    virtual std::complex<double> get_wave_function_at(double x, double y, double z, double t) const = 0;

    // Calculates overlap integral or cosine similarity with another agent
    virtual double calculate_overlap_with(const Agent& other) const = 0;

    // Returns the N-dimensional features for spatial partitioning (e.g. Barnes-Hut)
    virtual Eigen::VectorXd get_spatial_features() const = 0;

    // Semantic payload for LLM/Tool interactions
    std::unordered_map<std::string, std::string> get_semantic_payload() const { return semantic_payload; }
    void set_semantic_payload(const std::unordered_map<std::string, std::string>& payload) { semantic_payload = payload; }
    void update_semantic_payload(const std::string& key, const std::string& value) { semantic_payload[key] = value; }

protected:
    std::unordered_map<std::string, std::string> semantic_payload;
};

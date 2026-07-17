#include "ContinuousSubstrate.hpp"
#include "MicroAgent.hpp"
#include <future>
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>
#include <mutex>

ContinuousSubstrate::ContinuousSubstrate() {}

void ContinuousSubstrate::inject_agent(std::shared_ptr<Agent> agent) {
    field_agents.push_back(agent);
}

double ContinuousSubstrate::calculate_free_energy() const {
    double total_free_energy = 0.0;
    size_t N = field_agents.size();
    if (N == 0) return 0.0;

    // 1. Sort agents by their 0th spatial feature dimension (O(N log N))
    std::vector<std::shared_ptr<Agent>> sorted_agents = field_agents;
    if (sorted_agents[0]->get_spatial_features().size() > 0) {
        std::sort(sorted_agents.begin(), sorted_agents.end(), [](const std::shared_ptr<Agent>& a, const std::shared_ptr<Agent>& b) {
            return a->get_spatial_features()(0) < b->get_spatial_features()(0);
        });
    }

    // Cluster size B ~ sqrt(N)
    size_t B = std::max<size_t>(1, std::sqrt(N));
    size_t num_clusters = (N + B - 1) / B;
    
    std::vector<Eigen::VectorXd> cluster_centroids(num_clusters);
    
    // Compute centroids
    for (size_t c = 0; c < num_clusters; ++c) {
        size_t start = c * B;
        size_t end = std::min(N, start + B);
        Eigen::VectorXd sum = sorted_agents[start]->get_spatial_features();
        for (size_t i = start + 1; i < end; ++i) {
            sum += sorted_agents[i]->get_spatial_features();
        }
        cluster_centroids[c] = sum / (end - start);
    }
    
    std::mutex energy_mutex;
    std::vector<std::future<double>> futures;

    // 2. Exact pairwise overlap WITHIN clusters
    for (size_t c = 0; c < num_clusters; ++c) {
        size_t start = c * B;
        size_t end = std::min(N, start + B);
        
        for (size_t i = start; i < end; ++i) {
            for (size_t j = i + 1; j < end; ++j) {
                futures.push_back(std::async(std::launch::async, [&, i, j]() {
                    double overlap = sorted_agents[i]->calculate_overlap_with(*sorted_agents[j]);
                    return (1.0 - overlap) - dissipation_delta; 
                }));
            }
        }
    }
    
    // 3. Approximate pairwise overlap BETWEEN clusters
    for (size_t c1 = 0; c1 < num_clusters; ++c1) {
        for (size_t c2 = c1 + 1; c2 < num_clusters; ++c2) {
            futures.push_back(std::async(std::launch::async, [&, c1, c2]() {
                double dist_sq = (cluster_centroids[c1] - cluster_centroids[c2]).squaredNorm();
                double overlap = std::exp(-0.5 * dist_sq); // Gaussian overlap of centroids
                
                size_t size1 = std::min(N, (c1+1)*B) - (c1*B);
                size_t size2 = std::min(N, (c2+1)*B) - (c2*B);
                
                return ((1.0 - overlap) - dissipation_delta) * (size1 * size2);
            }));
        }
    }

    for (auto& f : futures) {
        total_free_energy += f.get();
    }

    return total_free_energy;
}

void ContinuousSubstrate::step_simulation(double dt) {
    // Advance the continuous simulation via gradient descent
    // dphi/dt = -nabla_phi F(phi)
    
    double current_energy = calculate_free_energy();
    double epsilon = 1e-4;
    double alpha = 0.05; // Learning rate for the phase descent
    
    std::vector<double> phase_gradients(field_agents.size(), 0.0);
    
    // 1. Calculate finite-difference gradients for phase (theta)
    for (size_t i = 0; i < field_agents.size(); ++i) {
        if (auto micro_agent = std::dynamic_pointer_cast<MicroAgent>(field_agents[i])) {
            double old_phase = micro_agent->get_phase();
            
            // Perturb forward
            micro_agent->set_phase(old_phase + epsilon);
            double new_energy = calculate_free_energy();
            
            // Numerical gradient
            phase_gradients[i] = (new_energy - current_energy) / epsilon;
            
            // Revert
            micro_agent->set_phase(old_phase);
        }
    }
    
    // 2. Apply gradient descent update
    for (size_t i = 0; i < field_agents.size(); ++i) {
        if (auto micro_agent = std::dynamic_pointer_cast<MicroAgent>(field_agents[i])) {
            double old_phase = micro_agent->get_phase();
            micro_agent->set_phase(old_phase - alpha * phase_gradients[i] * dt);
        }
    }
}

#ifndef MQI_CONFIG_HPP
#define MQI_CONFIG_HPP

/// \file
///
/// Configuration constants and parameters for Moqui Coarse2Fine simulation.
///
/// This header contains centralized configuration values that were previously
/// hardcoded throughout the codebase. Moving them here improves maintainability
/// and allows for easier tuning of simulation parameters.

namespace mqi {

/// Persistent Thread Pool Configuration
namespace threads {
constexpr uint32_t MIN_THREADS = 32;   ///< Minimum number of threads to create
constexpr uint32_t MAX_THREADS = 1024; ///< Maximum number of threads allowed
constexpr uint32_t THREADS_PER_BLOCK = 128; ///< Threads per CUDA block
constexpr float IDLE_TIMEOUT_MS =
    100.0f; ///< Thread idle timeout in milliseconds
} // namespace threads

/// Work Queue and Performance Configuration
namespace performance {
constexpr uint32_t TARGET_WORK_CHUNK_SIZE = 10; ///< Target work items per chunk
constexpr uint32_t WORK_QUEUE_CAPACITY =
    100000; ///< Maximum work queue capacity
constexpr bool WORK_STEALING_ENABLED =
    true; ///< Enable work stealing between threads
constexpr bool DYNAMIC_THREAD_SCALING = true; ///< Enable dynamic thread scaling
constexpr bool PERFORMANCE_MONITORING = true; ///< Enable performance monitoring
constexpr bool RNG_STANDARDIZATION = true;    ///< Use standardized RNG
} // namespace performance

/// Dynamic Level of Detail (LOD) Configuration
namespace lod {
/// Default distance thresholds for LOD switching (in mm)
constexpr float DEFAULT_DISTANCE_THRESHOLD_COARSE =
    500.0f; ///< Distance to switch to coarse LOD
constexpr float DEFAULT_DISTANCE_THRESHOLD_MEDIUM =
    100.0f; ///< Distance to switch to medium LOD

/// Minimum distance thresholds to prevent overly aggressive LOD switching
constexpr float MIN_DISTANCE_THRESHOLD_COARSE =
    50.0f; ///< Minimum coarse threshold
constexpr float MIN_DISTANCE_THRESHOLD_MEDIUM =
    200.0f; ///< Minimum medium threshold

/// Performance tuning parameters
constexpr float DEFAULT_TARGET_PERFORMANCE_MS =
    16.67f; ///< Target frame time (60 FPS)
constexpr float COST_SCALE_FACTOR =
    100.0f; ///< Scale factor for cost calculations
} // namespace lod

/// Physics and Simulation Tolerance Configuration
namespace physics {
constexpr float GEOMETRY_TOLERANCE =
    1e-3f;                         ///< Tolerance for geometric calculations
constexpr float NEAR_ZERO = 1e-7f; ///< Threshold for considering values as zero
constexpr float MIN_STEP_SIZE =
    1e-3f; ///< Minimum step size for particle transport
} // namespace physics

/// Memory and Resource Configuration
namespace memory {
constexpr size_t DEFAULT_CACHE_SIZE = 1024 * 1024; ///< Default cache size (1MB)
constexpr uint32_t MAX_ALLOCATIONS = 10000; ///< Maximum concurrent allocations
} // namespace memory

} // namespace mqi

#endif // MQI_CONFIG_HPP

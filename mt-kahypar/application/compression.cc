#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>
namespace fs = std::filesystem;

#include <iostream>

#include "mt-kahypar/definitions.h"
#include "mt-kahypar/io/command_line_options.h"
#include "mt-kahypar/io/hypergraph_factory.h"
#include "mt-kahypar/io/partitioning_output.h"
#include "mt-kahypar/io/presets.h"
#include "mt-kahypar/partition/conversion.h"
#include "mt-kahypar/partition/mapping/target_graph.h"
#include "mt-kahypar/partition/partitioner_facade.h"
#include "mt-kahypar/partition/registries/register_memory_pool.h"
#include "mt-kahypar/partition/registries/registry.h"
#include "mt-kahypar/utils/cast.h"
#include "mt-kahypar/utils/delete.h"
#include "mt-kahypar/utils/exception.h"
#include "mt-kahypar/utils/randomize.h"
#include "mt-kahypar/utils/utilities.h"

using namespace mt_kahypar;

void write_csv_header(const char* csv_path) {
    std::ofstream outfile(csv_path, std::ios::trunc);  // Overwrite if exists
    outfile
        << "FileName,NodeCount,HyperedgeCount,IOTimeMS,MemoryUsage,Compressed"
        << std::endl;
}

void append_csv_line(const char* csv_path, const char* filename,
                     size_t num_nodes, size_t num_hyperedges, long long io_time,
                     size_t memory_usage, bool compressed) {
    std::ofstream outfile(csv_path, std::ios::app);
    outfile << filename << "," << num_nodes << "," << num_hyperedges << ","
            << io_time << "," << memory_usage << "," << (compressed ? "1" : "0")
            << std::endl;
}

void write_graph() {}

void read_hypergraph(const char* path, const char* OUTPUT_PATH,
                     bool compressed = false) {
    auto start = std::chrono::high_resolution_clock::now();

    // COMPRESSION_TODO
    // mt_kahypar_hypergraph_t hypergraph = io::readInputFile(
    //     context.partition.graph_filename, context.partition.preset_type,
    //     context.partition.instance_type, context.partition.file_format,
    //     context.preprocessing.stable_construction_of_incident_edges);
    std::cout << "Using compressed hypergraph format." << std::endl;
    mt_kahypar_hypergraph_t hypergraph =
        io::readInputFile(path, PresetType::default_preset,
                          compressed ? InstanceType::compressed_hypergraph
                                     : InstanceType::hypergraph,
                          FileFormat::hMetis, false);
    auto end = std::chrono::high_resolution_clock::now();
    long long io_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();

    if (compressed) {
        const ds::CompressedHypergraph& hg =
            utils::cast<ds::CompressedHypergraph>(hypergraph);
        append_csv_line(OUTPUT_PATH, path, hg.initialNumNodes(),
                        hg.initialNumEdges(), io_time, hg.memoryConsumptionKB(),
                        compressed);
    } else {
        const ds::StaticHypergraph& hg =
            utils::cast<ds::StaticHypergraph>(hypergraph);
        append_csv_line(OUTPUT_PATH, path, hg.initialNumNodes(),
                        hg.initialNumEdges(), io_time, hg.memoryConsumptionKB(),
                        compressed);
    }

    // std::vector<size_t> partition;
    //     mt_kahypar_error_t error{};

    //     mt_kahypar_initialize(
    //         std::thread::hardware_concurrency(),
    //         true
    //     );

    //     mt_kahypar_context_t* context =
    //     mt_kahypar_context_from_preset(DEFAULT);
    //     mt_kahypar_set_partitioning_parameters(context, 3, 0.03, CUT);
    //     mt_kahypar_set_seed(42);
    //     mt_kahypar_set_context_parameter(context, VERBOSE, "0", &error);

    //     mt_kahypar_hypergraph_t hypergraph =  compressed
    //         ? mt_kahypar_stream_hypergraph_from_file(path, context, HMETIS,
    //         &error) : mt_kahypar_read_hypergraph_from_file(path, context,
    //         HMETIS, &error);
    //     // io::readInputFile(file_name, context.partition.preset_type,
    //     instance_type, file_format, true) if (hypergraph.hypergraph ==
    //     nullptr) {
    //         std::cout << error.msg << std::endl;
    //         std::exit(1);
    //     }

    //     size_t num_nodes = mt_kahypar_num_hypernodes(hypergraph);
    //     size_t num_hyperedges = mt_kahypar_num_hyperedges(hypergraph);
    //     // size_t num_pins = mt_kahypar_num_pins(hypergraph);
    //     // size_t total_weight = mt_kahypar_hypergraph_weight(hypergraph);

    //     auto end = std::chrono::high_resolution_clock::now();
    //     long long partition_time =
    //     std::chrono::duration_cast<std::chrono::milliseconds>(end -
    //     start).count();

    //     size_t memory_usage = mt_kahypar_memory_kb(hypergraph);

    //     append_csv_line(OUTPUT_PATH, path, num_nodes, num_hyperedges,
    //     partition_time, memory_usage, compressed);

    //     mt_kahypar_free_context(context);
    //     mt_kahypar_free_hypergraph(hypergraph);
}

void print_progress(size_t current, size_t total) {
    const int bar_width = 50;  // Width of the progress bar
    float progress = static_cast<float>(current) / total;

    std::cout << "\r[";
    int pos = static_cast<int>(bar_width * progress);
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos)
            std::cout << "=";
        else if (i == pos)
            std::cout << ">";
        else
            std::cout << " ";
    }
    std::cout << "] " << current << "/" << total << " files" << std::flush;
}

int main(int argc, char* argv[]) {
    Context context(false);
    processCommandLineInput(context, argc, argv, nullptr);

    if (context.partition.preset_file == "") {
        if (context.partition.preset_type != PresetType::UNDEFINED)
            throw InvalidInputException("No preset specified");
        // Only a preset type specified => load according preset
        auto preset_option_list = loadPreset(context.partition.preset_type);
        processCommandLineInput(context, argc, argv, &preset_option_list);
    }

    // Determine instance (graph or hypergraph) and partition type
    if (context.partition.instance_type == InstanceType::UNDEFINED) {
        context.partition.instance_type =
            to_instance_type(context.partition.file_format);
    }
    context.partition.partition_type = to_partition_c_type(
        context.partition.preset_type, context.partition.instance_type);

    context.utility_id =
        utils::Utilities::instance().registerNewUtilityObjects();
    if (context.partition.verbose_output) {
        io::printBanner();
    }

    utils::Randomize::instance().setSeed(context.partition.seed);
    if (context.shared_memory.use_localized_random_shuffle) {
        utils::Randomize::instance().enableLocalizedParallelShuffle(
            context.shared_memory.shuffle_block_size);
    }

#ifndef KAHYPAR_DISABLE_HWLOC
    size_t num_available_cpus = HardwareTopology::instance().num_cpus();
    if (num_available_cpus < context.shared_memory.num_threads) {
        WARNING("There are currently only"
                << num_available_cpus << "cpus available."
                << "Setting number of threads from"
                << context.shared_memory.num_threads << "to"
                << num_available_cpus);
        context.shared_memory.num_threads = num_available_cpus;
    }
#endif

    // Initialize TBB task arenas on numa nodes
    TBBInitializer::instance(context.shared_memory.num_threads);

#ifndef KAHYPAR_DISABLE_HWLOC
    // We set the membind policy to interleaved allocations in order to
    // distribute allocations evenly across NUMA nodes
    hwloc_cpuset_t cpuset = TBBInitializer::instance().used_cpuset();
    parallel::HardwareTopology<>::instance()
        .activate_interleaved_membind_policy(cpuset);
    hwloc_bitmap_free(cpuset);
#endif

    const char* OUTPUT_PATH = "benchmark.csv";
    write_csv_header(OUTPUT_PATH);
    read_hypergraph("./_graphs/benchmark_set_d/vibrobox.mtx.hgr", OUTPUT_PATH,
                    true);
    read_hypergraph("./_graphs/benchmark_set_d/vibrobox.mtx.hgr", OUTPUT_PATH,
                    false);

    //   const char* OUTPUT_PATH = "benchmark.csv";
    // write_csv_header(OUTPUT_PATH);

    // const std::string directory = "./benchmark_set_b";

    // // Collect all regular files first
    // std::vector<fs::path> files;
    // for (const auto& entry : fs::directory_iterator(directory)) {
    //     if (fs::is_regular_file(entry)) {
    //         files.push_back(entry.path());
    //     }
    // }

    // size_t total = files.size();
    // if (total == 0) {
    //     std::cout << "No files found in " << directory << std::endl;
    //     return 1;
    // }

    // size_t count = 0;
    // for (const auto& path : files) {
    //     ++count;
    //     read_hypergraph(path.string().c_str(), OUTPUT_PATH);
    //     read_hypergraph(path.string().c_str(), OUTPUT_PATH, true);
    //     print_progress(count, total);
    // }

    // std::cout << std::endl << "Processing complete." << std::endl;
    return 0;
}
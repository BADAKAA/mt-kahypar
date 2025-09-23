#include "compressed_hypergraph.h"
#include "compressed_hypergraph_factory.h"
#include "mt-kahypar/datastructures/compressed_hypergraph.h"
#include "mt-kahypar/utils/exception.h"
#include <fstream>
#include <charconv>
#include <chrono>
#include <cctype>
#include <algorithm>


namespace mt_kahypar::ds {

static inline bool is_space(char c) {
  return std::isspace(static_cast<unsigned char>(c)) != 0;
}

/**
 * Parses all unsigned numbers from a line (no index shifting)
 * 
 * The exclude_first flag is used to extract edge weights
 */
static inline size_t parseRawNumbers(const std::string& line, std::vector<size_t>& out, bool exclude_first = false) {
  out.clear();
  const char* ptr = line.c_str();
  const char* end = ptr + line.size();
  size_t first = 0;
  bool first_extracted = !exclude_first;
  while (ptr < end) {
    while (ptr < end && (is_space(*ptr) || *ptr == ',')) ++ptr;
    if (ptr >= end) break;
    size_t v = 0;
    auto [p, ec] = std::from_chars(ptr, end, v);
    ptr = p;
    if (ec != std::errc()) {
      throw InvalidInputException("Invalid number in input line: " + (line.size() <= 40 ? line : line.substr(0, 40) + "..."));
    }
    if (!first_extracted) {
      first = v;
      first_extracted = true;
      continue;
    } 
    out.push_back(v);
  }
  if (!first_extracted) throw InvalidInputException("Expected first value in input line but found none");
  return first;
}

// returns the line count of the header
size_t parseHeader(std::ifstream& in, HypernodeID& num_nodes, HyperedgeID& num_he, mt_kahypar::Type& type) {
  type = mt_kahypar::Type::Unweighted; // default
  std::string line;
  size_t line_number = 0;

  while (std::getline(in, line)) {
    ++line_number;
    std::string trimmed = line;
    auto it = std::find_if_not(trimmed.begin(), trimmed.end(), [](unsigned char c){ return std::isspace(c); });
    if (it == trimmed.end()) continue;
    if (*it == '%' || *it == '#') continue;

    std::vector<size_t> nums;
    parseRawNumbers(trimmed, nums);
    if (nums.size() < 2) throw InvalidInputException("Invalid header in input file line: " + std::to_string(line_number));
    num_he = static_cast<HyperedgeID>(nums[0]);
    num_nodes = static_cast<HypernodeID>(nums[1]);
    if (nums.size() > 2) type = static_cast<mt_kahypar::Type>(nums[2]);
    if (nums.size() > 3) throw InvalidInputException("Invalid header in input file line: " + std::to_string(line_number));
    break;
  }
  if (num_nodes == 0 && num_he == 0) throw InvalidInputException("Input file is missing header");
  return line_number;
}

// Write a single value as unsigned LEB128 (7-bit groups, high-bit is continuation)
static inline size_t push_varint(std::vector<uint8_t>& out, size_t x) {
  size_t blocks = 1;
  while (x >= 0x80u) {
    out.push_back(static_cast<uint8_t>((x & 0x7Fu) | 0x80u));
    x >>= 7;
    ++blocks;
  }
  out.push_back(static_cast<uint8_t>(x & 0x7Fu));
  return blocks;
}

size_t get_last_varint(std::vector<uint8_t>& vec) {
  if (vec.empty()) return 0;
  size_t pos = vec.size() - 1;
  size_t result = 0;
  int shift = 0;
  while (true) {
    uint8_t byte = vec[pos];
    result |= (static_cast<size_t>(byte & 0x7Fu) << shift);
    if ((byte & 0x80u) == 0) break;
    if (pos == 0) throw InvalidInputException("Malformed varint in vector");
    --pos;
    shift += 7;
  }
  return result;
}

CompressedHypergraph CompressedHypergraphFactory::stream(const std::string& filename, const bool remove_single_pin_hes) {
  using Hypernode = CompressedHypergraph::Hypernode;

  std::ifstream in(filename);
  if (!in) throw mt_kahypar::SystemException("Cannot open file: " + filename);

  HypernodeID n_vertices = 0;
  HyperedgeID n_edges = 0;
  mt_kahypar::Type type;

  size_t current_line = parseHeader(in, n_vertices, n_edges, type);
  
  const HypernodeID V = n_vertices;
  const HyperedgeID H = n_edges;

  CompressedHypergraph hg = CompressedHypergraph();

  hg._num_hypernodes = V;
  hg._num_pins       = 0;
  hg._num_hyperedges = 0;
  hg._num_removed_hyperedges = 0;
  hg._max_edge_size = 0;
  hg._total_degree = 0;
  hg._total_weight = 0;

  hg._hypernodes.resize(V);
  hg._hyperedges.clear();
  hg._hyperedges.reserve(H);

  for (Hypernode& hn : hg._hypernodes) hn.enable();

  const bool has_edge_weights = (type == mt_kahypar::Type::EdgeWeights || type == mt_kahypar::Type::EdgeAndNodeWeights);
  const bool has_node_weights = (type == mt_kahypar::Type::NodeWeights || type == mt_kahypar::Type::EdgeAndNodeWeights);

  std::string line;
  std::vector<size_t> nums;

  // Build incident nets per node directly in compressed form using monotonic current_he_id
  std::vector<std::vector<uint8_t>> incident_nets(V);
  std::vector<HyperedgeID> last_he_for_node(V, 0);
  size_t compressed_incident_nets_size = 0;

  HyperedgeID current_he_id = 0; //  == number of valid hyperedges after loop
  HyperedgeID removed_single_pin = 0;

  while (std::getline(in, line)) {
    ++current_line;
    if (line.empty()) continue;

    size_t he_weight = parseRawNumbers(line, nums, has_edge_weights);
    if (nums.empty()) continue;

    std::sort(nums.begin(), nums.end());
    nums.erase(std::unique(nums.begin(), nums.end()), nums.end());

    if (remove_single_pin_hes && nums.size() <= 1) {
      ++removed_single_pin;
      continue;
    }

    size_t start = hg._compressed_incidence_array.size();
    size_t max_id = static_cast<size_t>(V);
    size_t prev = 0;
    for (size_t i = 0; i < nums.size(); ++i) {
      size_t cur = nums[i] - 1;
      if (cur >= max_id) {
        throw InvalidInputException("Node ID '" +  std::to_string(cur + 1) + "' out of bounds in line " + std::to_string(current_line));
      }
      size_t gap = cur - prev;
      push_varint(hg._compressed_incidence_array, gap);
  const size_t net_gap = static_cast<size_t>(current_he_id - last_he_for_node[cur]);
  compressed_incident_nets_size += push_varint(incident_nets[cur], net_gap);
  last_he_for_node[cur] = current_he_id;
      hg._hypernodes[cur].setUncompressedSize(hg._hypernodes[cur].size() + 1);
      prev = cur;
    }

    hg._hyperedges.emplace_back();
    auto& he = hg._hyperedges.back();
    he.enable();
    he.setFirstEntry(start);
    he.setUncompressedSize(nums.size());
    he.setCompressedSize(hg._compressed_incidence_array.size() - start);
    if (has_edge_weights) he.setWeight(static_cast<HyperedgeWeight>(he_weight));

    hg._num_pins += static_cast<HypernodeID>(nums.size());
    if (nums.size() > static_cast<size_t>(hg._max_edge_size)) {
      hg._max_edge_size = static_cast<HypernodeID>(nums.size());
    }

    ++current_he_id;
  }

  hg._num_hyperedges = current_he_id;
  hg._num_removed_hyperedges = removed_single_pin;

  // Read node weights if present (V weights as raw numbers possibly across lines)
  std::vector<HypernodeWeight> node_weights;
  if (has_node_weights) {
    node_weights.reserve(V);
    while (node_weights.size() < static_cast<size_t>(V) && std::getline(in, line)) {
      ++current_line;
      auto it2 = std::find_if_not(line.begin(), line.end(), [](unsigned char c){ return std::isspace(c); });
      if (it2 == line.end() || *it2 == '%' || *it2 == '#') continue;
      parseRawNumbers(line, nums);
      for (size_t w : nums) {
        if (node_weights.size() < static_cast<size_t>(V)) node_weights.push_back(static_cast<HypernodeWeight>(w));
      }
    }
    if (node_weights.size() != static_cast<size_t>(V)) {
      throw InvalidInputException("Not enough node weights in weighted HGR; expected " + std::to_string(V));
    }
  }

  // combine all per-node compressed incident nets into CSR representation
  hg._compressed_incident_nets.clear();
  hg._compressed_incident_nets.reserve(compressed_incident_nets_size);

  hg._total_degree = 0;
  hg._total_weight = 0;
   for (size_t id = 0; id < incident_nets.size(); ++id) {
    std::vector<uint8_t>& arr = incident_nets[id];
    Hypernode& hn = hg._hypernodes[id];
    hg._total_degree += static_cast<HypernodeID>(hn.size());
    hn.setCompressedSize(arr.size());
    hn.setFirstEntry(hg._compressed_incident_nets.size());
    hg._compressed_incident_nets.insert(
        hg._compressed_incident_nets.end(),
        std::make_move_iterator(arr.begin()),
        std::make_move_iterator(arr.end())
    );
    arr.clear();

    if (!has_node_weights) continue;
    HypernodeWeight w = node_weights[id];
    hn.setWeight(w);
    hg._total_weight += w;
  }
  if (!has_node_weights) hg._total_weight = static_cast<HypernodeWeight>(V);

  hg._community_ids.assign(hg._num_hypernodes, 0);
  return hg;
  }

CompressedHypergraph CompressedHypergraphFactory::construct(
    const HypernodeID num_hypernodes,
    const HyperedgeID /* num_hyperedges */, // derived from edge_vector.size()
    const HyperedgeVector& edge_vector,
    const HyperedgeWeight* hyperedge_weight,
    const HypernodeWeight* hypernode_weight,
    const bool /* stable_construction_of_incident_edges */) {

  using Hypernode = CompressedHypergraph::Hypernode;

  const HypernodeID V = num_hypernodes;
  const HyperedgeID H = static_cast<HyperedgeID>(edge_vector.size());

  CompressedHypergraph hg;

  // Initialize sizes and counters
  hg._num_hypernodes = V;
  hg._num_hyperedges = H;
  hg._num_removed_hyperedges = 0;
  hg._max_edge_size = 0;
  hg._num_pins = 0;
  hg._total_degree = 0;
  hg._total_weight = 0;

  // Allocate containers
  hg._hypernodes.resize(V);
  hg._hyperedges.resize(H);
  hg._compressed_incidence_array.clear();
  hg._compressed_incident_nets.clear();

  // Enable nodes and set initial weights
  for (HypernodeID u = 0; u < V; ++u) {
    auto& hn = hg._hypernodes[u];
    hn.enable();
    if (hypernode_weight) {
      hn.setWeight(hypernode_weight[u]);
      hg._total_weight += hypernode_weight[u];
    } else {
      // default weight is 1 per constructor; accumulate explicitly
      hg._total_weight += 1;
    }
  }

  // Build incident nets per node directly in compressed form (varint gap-encoded).
  std::vector<std::vector<uint8_t>> incident_nets(V);
  std::vector<HyperedgeID> last_he_for_node(V, 0);
  size_t total_incident_bytes_est = 0;

  // Encode edges' pin lists into one contiguous compressed buffer
  size_t pins_bytes_offset = 0;
  for (HyperedgeID he = 0; he < H; ++he) {
    const auto& pins_in = edge_vector[he];

    // Validate pins and prepare a sorted, deduplicated pin list (0-based IDs expected)
    std::vector<HypernodeID> pins;
    pins.reserve(pins_in.size());
    for (HypernodeID v : pins_in) {
      if (v >= V) {
        throw InvalidInputException("Hypernode ID out of bounds in construct(): " + std::to_string(v));
      }
      pins.push_back(v);
    }
    std::sort(pins.begin(), pins.end());
    pins.erase(std::unique(pins.begin(), pins.end()), pins.end());

    // Create and enable hyperedge
    auto& out_he = hg._hyperedges[he];
    out_he.enable();
    out_he.setFirstEntry(pins_bytes_offset);
    out_he.setUncompressedSize(pins.size());
    if (hyperedge_weight) {
      out_he.setWeight(hyperedge_weight[he]);
    }

    // Varint gap-encode pins into incidence array
    HypernodeID prev = 0;
    for (HypernodeID v : pins) {
      const uint64_t gap = static_cast<uint64_t>(v - prev);
      encode_varint(gap, hg._compressed_incidence_array);
      prev = v;
      // Append compressed gap for incident net of node v (he is increasing)
      const size_t net_gap = static_cast<size_t>(he - last_he_for_node[v]);
      total_incident_bytes_est += push_varint(incident_nets[v], net_gap);
      last_he_for_node[v] = he;
      // Track degree counts
      hg._hypernodes[v].setUncompressedSize(hg._hypernodes[v].size() + 1);
    }
    const size_t new_end = hg._compressed_incidence_array.size();
    const size_t written = new_end - pins_bytes_offset;
    out_he.setCompressedSize(written);
    pins_bytes_offset = new_end;

    // Global counters
    hg._num_pins += static_cast<HypernodeID>(pins.size());
    if (pins.size() > static_cast<size_t>(hg._max_edge_size)) {
      hg._max_edge_size = static_cast<HypernodeID>(pins.size());
    }
  }

  // Combine per-node compressed sequences into CSR-like single array
  hg._compressed_incident_nets.clear();
  hg._compressed_incident_nets.reserve(total_incident_bytes_est);
  size_t nets_bytes_offset = 0;
  for (HypernodeID u = 0; u < V; ++u) {
    Hypernode& hn = hg._hypernodes[u];
    std::vector<uint8_t>& bytes = incident_nets[u];
    hn.setFirstEntry(nets_bytes_offset);
    hn.setCompressedSize(bytes.size());
    hg._compressed_incident_nets.insert(
      hg._compressed_incident_nets.end(),
      std::make_move_iterator(bytes.begin()),
      std::make_move_iterator(bytes.end())
    );
    nets_bytes_offset += hn.compressedSize();
    hg._total_degree += static_cast<HypernodeID>(hn.size());
    bytes.clear();
  }

  // Communities default to 0
  hg._community_ids.assign(hg._num_hypernodes, 0);
  return hg;
}
}

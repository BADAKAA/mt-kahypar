/*******************************************************************************
 * MIT License
 *
 * This file is part of Mt-KaHyPar.
 *
 * Copyright (C) 2019 Lars Gottesbüren <lars.gottesbueren@kit.edu>
 * Copyright (C) 2019 Tobias Heuer <tobias.heuer@kit.edu>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 ******************************************************************************/

#include "compressed_hypergraph.h"

#include "mt-kahypar/parallel/parallel_prefix_sum.h"
#include "mt-kahypar/datastructures/concurrent_bucket_map.h"
#include "mt-kahypar/utils/timer.h"
#include "mt-kahypar/utils/memory_tree.h"

#include <tbb/parallel_reduce.h>
#include <tbb/parallel_sort.h>

#include <unordered_map>
#include <algorithm>
#include <numeric>

namespace mt_kahypar::ds {


  /*!
  * This struct is used during multilevel coarsening to efficiently
  * detect parallel hyperedges.
  */
  struct ContractedHyperedgeInformation {
    HyperedgeID he = kInvalidHyperedge;
    size_t hash = kEdgeHashSeed;
    size_t size = std::numeric_limits<size_t>::max();
    bool valid = false;
  };

  /*!
   * Contracts a given community structure. All vertices with the same label
   * are collapsed into the same vertex. The resulting single-pin and parallel
   * hyperedges are removed from the contracted graph. The function returns
   * the contracted hypergraph and a mapping which specifies a mapping from
   * community label (given in 'communities') to a vertex in the coarse hypergraph.
   *
   * \param communities Community structure that should be contracted
   */
  CompressedHypergraph CompressedHypergraph::contract(parallel::scalable_vector<HypernodeID>& communities, bool deterministic) {
    // Stage 0: Preconditions and helpers
    ASSERT(communities.size() == _num_hypernodes);

    auto map_to_coarse = [&](const HypernodeID hn) -> HypernodeID {
      ASSERT(hn < communities.size());
      return communities[hn];
    };

    // #################### STAGE 1 ####################
    // Densify community IDs and compute number of coarse nodes
    // We assume communities[hn] in [0, _num_hypernodes), but only enabled nodes count.
    std::vector<uint8_t> present(_num_hypernodes, 0);
    for (HypernodeID hn = 0; hn < _num_hypernodes; ++hn) {
      if (nodeIsEnabled(hn)) {
        const HypernodeID cid = communities[hn];
        ASSERT(cid < _num_hypernodes);
        present[cid] = 1;
      } else {
        communities[hn] = kInvalidHypernode;
      }
    }

    std::vector<HypernodeID> label_to_coarse(_num_hypernodes, kInvalidHypernode);
    HypernodeID next_coarse = 0;
    for (HypernodeID lab = 0; lab < _num_hypernodes; ++lab) {
      if (present[lab]) label_to_coarse[lab] = next_coarse++;
    }
    const HypernodeID num_coarse_nodes = next_coarse;

    // Remap communities to dense IDs
    for (HypernodeID hn = 0; hn < _num_hypernodes; ++hn) {
      if (communities[hn] != kInvalidHypernode) {
        communities[hn] = label_to_coarse[communities[hn]];
      }
    }

    // Aggregate coarse node weights
    std::vector<HypernodeWeight> coarse_node_weight(num_coarse_nodes, 0);
    for (HypernodeID hn = 0; hn < _num_hypernodes; ++hn) {
      const HypernodeID ch = communities[hn];
      if (ch != kInvalidHypernode) {
        coarse_node_weight[ch] += nodeWeight(hn);
      }
    }

    // #################### STAGE 2 ####################
    // Remap hyperedges to coarse graph, remove single-pin nets, and prepare for parallel merging
    struct VectorHash {
      size_t operator()(const std::vector<HypernodeID>& v) const noexcept {
        // 64-bit mix (similar to boost::hash_combine)
        size_t h = 1469598103934665603ull;
        for (HypernodeID x : v) {
          h ^= static_cast<size_t>(x) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        }
        return h;
      }
    };

    std::unordered_map<std::vector<HypernodeID>, HyperedgeID, VectorHash> parallel_map;
    parallel_map.reserve(static_cast<size_t>(_num_hyperedges));

    std::vector<std::vector<HypernodeID>> coarse_edges;       // pins per coarse edge (sorted, unique)
    coarse_edges.reserve(static_cast<size_t>(_num_hyperedges));
    std::vector<HyperedgeWeight> coarse_edge_weight;          // merged weights
    coarse_edge_weight.reserve(static_cast<size_t>(_num_hyperedges));

    // Decode every enabled edge, map to coarse, sort+unique, drop single-pin,
    // and merge parallel hyperedges by identical pin sets.
    for (HyperedgeID he = 0; he < _num_hyperedges; ++he) {
      if (!edgeIsEnabled(he)) continue;

      const size_t begin = _hyperedge_offsets[he];
      const size_t end   = hyperedge_firstInvalidEntry(he);
      size_t pos = begin;
      HypernodeID prev = 0;

      std::vector<HypernodeID> pins;
      // Read header varint for uncompressed edge size and reserve
      const size_t esize = static_cast<size_t>(decode_varint_bounded(_compressed_incidence_array, pos, end));
      pins.reserve(esize);
      size_t emitted = 0;
      while (pos < end && emitted < esize) {
        const uint64_t gap = decode_varint_bounded(_compressed_incidence_array, pos, end);
        if (emitted > 0 && gap == 0) { continue; }
        const HypernodeID v = static_cast<HypernodeID>(prev + gap);
        prev = v;
        ++emitted;
        const HypernodeID ch = map_to_coarse(v);
        if (ch != kInvalidHypernode) {
          pins.push_back(ch);
        }
      }
      if (pins.empty()) continue;

      std::sort(pins.begin(), pins.end());
      pins.erase(std::unique(pins.begin(), pins.end()), pins.end());

      if (pins.size() <= 1) {
        // becomes single-pin in coarse graph => drop
        continue;
      }

      auto it = parallel_map.find(pins);
      if (it == parallel_map.end()) {
        HyperedgeID new_id = static_cast<HyperedgeID>(coarse_edges.size());
        parallel_map.emplace(pins, new_id);
        coarse_edge_weight.push_back(edgeWeight(he));
        coarse_edges.emplace_back(std::move(pins));
      } else {
        // parallel net: accumulate weights
        coarse_edge_weight[it->second] += edgeWeight(he);
      }
    }

    // Determinism: optional reordering by pins to stabilize IDs
    // We keep insertion order for performance; if requested, sort by pins lexicographically
    if (deterministic) {
      // Build permutation by comparing pin vectors
      std::vector<HyperedgeID> perm(coarse_edges.size());
      std::iota(perm.begin(), perm.end(), 0);
      std::stable_sort(perm.begin(), perm.end(), [&](HyperedgeID a, HyperedgeID b) {
        const auto& A = coarse_edges[a];
        const auto& B = coarse_edges[b];
        return std::lexicographical_compare(A.begin(), A.end(), B.begin(), B.end());
      });
      // Apply permutation
      std::vector<std::vector<HypernodeID>> edges_sorted;
      edges_sorted.reserve(coarse_edges.size());
      std::vector<HyperedgeWeight> weights_sorted;
      weights_sorted.reserve(coarse_edges.size());
      std::vector<HyperedgeID> old_to_new(coarse_edges.size());
      for (HyperedgeID new_id = 0; new_id < perm.size(); ++new_id) {
        edges_sorted.emplace_back(std::move(coarse_edges[perm[new_id]]));
        weights_sorted.emplace_back(coarse_edge_weight[perm[new_id]]);
        old_to_new[perm[new_id]] = new_id;
      }
      coarse_edges = std::move(edges_sorted);
      coarse_edge_weight = std::move(weights_sorted);
      // No need to update parallel_map further, we won't use it anymore.
      (void)old_to_new;
    }

    // #################### STAGE 3 ####################
    // Build coarse hypergraph with compressed storage
    CompressedHypergraph hypergraph;
    hypergraph._num_hypernodes = num_coarse_nodes;
    hypergraph._num_hyperedges = static_cast<HyperedgeID>(coarse_edges.size());
    hypergraph._num_removed_hyperedges = 0;
    hypergraph._max_edge_size = 0;
    hypergraph._num_pins = 0;
    hypergraph._total_degree = 0;
    hypergraph._total_weight = 0;

    // Init CSR containers
    hypergraph._hypernode_offsets.assign(num_coarse_nodes, 0);
    hypergraph._hypernode_enabled.assign(num_coarse_nodes, 1);
    hypergraph._hyperedge_offsets.assign(hypergraph._num_hyperedges, 0);
    hypergraph._hyperedge_enabled.assign(hypergraph._num_hyperedges, 1);
    hypergraph._compressed_incidence_array.clear();
    hypergraph._compressed_incident_nets.clear();
    // Initialize lazy weights
    hypergraph._hypernode_weights.assign(num_coarse_nodes, 1);
    hypergraph._hyperedge_weights.assign(hypergraph._num_hyperedges, 1);

    // Set node weights; prepare incident nets buckets
    std::vector<std::vector<HyperedgeID>> incident_nets(num_coarse_nodes);
    for (HypernodeID u = 0; u < num_coarse_nodes; ++u) {
      hypergraph._hypernode_weights[u] = coarse_node_weight[u];
      // firstEntry/compressed sizes set later
    }

    // Encode coarse edges (pins) and collect incident nets
    size_t pins_bytes_offset = 0;
    for (HyperedgeID he = 0; he < hypergraph._num_hyperedges; ++he) {
      const auto& pins = coarse_edges[he];
      hypergraph._hyperedge_offsets[he] = pins_bytes_offset;
      // header varint for pins size
      encode_varint(static_cast<uint64_t>(pins.size()), hypergraph._compressed_incidence_array);
      hypergraph._hyperedge_weights[he] = coarse_edge_weight[he];

      // Varint gap-encode pins
      HypernodeID prev = 0;
      for (HypernodeID v : pins) {
        uint64_t gap = static_cast<uint64_t>(v - prev);
        encode_varint(gap, hypergraph._compressed_incidence_array);
        prev = v;
      }
  const size_t new_end = hypergraph._compressed_incidence_array.size();
  pins_bytes_offset = new_end;

      hypergraph._num_pins += static_cast<HypernodeID>(pins.size());
      if (pins.size() > static_cast<size_t>(hypergraph._max_edge_size)) {
        hypergraph._max_edge_size = static_cast<HypernodeID>(pins.size());
      }
      // Fill incident nets buckets
      for (HypernodeID v : pins) {
        incident_nets[v].push_back(he);
      }
    }

    // Encode incident nets per node
    HypernodeWeight tw = 0;
    for (HypernodeID u = 0; u < num_coarse_nodes; ++u) {
      auto& list = incident_nets[u];

      std::sort(list.begin(), list.end());
      list.erase(std::unique(list.begin(), list.end()), list.end());

  // Start position for this node's incident nets
  hypergraph._hypernode_offsets[u] = hypergraph._compressed_incident_nets.size();
      // header varint for degree
      encode_varint(static_cast<uint64_t>(list.size()), hypergraph._compressed_incident_nets);

      HyperedgeID prev_e = 0;
      for (HyperedgeID e : list) {
        uint64_t gap = static_cast<uint64_t>(e - prev_e);
        encode_varint(gap, hypergraph._compressed_incident_nets);
        prev_e = e;
      }

      hypergraph._total_degree += static_cast<HypernodeID>(list.size());
      tw += hypergraph._hypernode_weights[u];
    }
    hypergraph._total_weight = tw;

    // #################### STAGE 4 ####################
    // Communities: inherit from fine graph (like static version)
    hypergraph._community_ids.assign(num_coarse_nodes, 0);
    for (HypernodeID hn = 0; hn < _num_hypernodes; ++hn) {
      const HypernodeID ch = communities[hn];
      if (ch != kInvalidHypernode) {
        hypergraph._community_ids[ch] = communityID(hn);
      }
    }

    // Fixed vertices
    if (hasFixedVertices()) {
      FixedVertexSupport<CompressedHypergraph> coarse_fixed(hypergraph.initialNumNodes(), _fixed_vertices.numBlocks());
      coarse_fixed.setHypergraph(&hypergraph);
      for (HypernodeID hn = 0; hn < _num_hypernodes; ++hn) {
        if (isFixed(hn)) {
          const HypernodeID ch = communities[hn];
          if (ch != kInvalidHypernode) {
            coarse_fixed.fixToBlock(ch, fixedVertexBlock(hn));
          }
        }
      }
      hypergraph.addFixedVertexSupport(std::move(coarse_fixed));
    }

    return hypergraph;
  }

  // ! Copy compressed hypergraph in parallel
  CompressedHypergraph CompressedHypergraph::copy(parallel_tag_t) const {
    CompressedHypergraph hypergraph;

    hypergraph._num_hypernodes = _num_hypernodes;
    hypergraph._num_removed_hypernodes = _num_removed_hypernodes;
    hypergraph._num_hyperedges = _num_hyperedges;
    hypergraph._num_removed_hyperedges = _num_removed_hyperedges;
    hypergraph._max_edge_size = _max_edge_size;
    hypergraph._num_pins = _num_pins;
    hypergraph._total_degree = _total_degree;
    hypergraph._total_weight = _total_weight;

    tbb::parallel_invoke([&] {
      hypergraph._hypernode_offsets = _hypernode_offsets;
    }, [&] {
      hypergraph._hypernode_enabled = _hypernode_enabled;
    }, [&] {
      hypergraph._compressed_incident_nets.resize(_compressed_incident_nets.size());
      memcpy(hypergraph._compressed_incident_nets.data(), _compressed_incident_nets.data(),
             sizeof(uint8_t) * _compressed_incident_nets.size());
    }, [&] {
      hypergraph._hyperedge_offsets = _hyperedge_offsets;
    }, [&] {
      hypergraph._hyperedge_enabled = _hyperedge_enabled;
    }, [&] {
      hypergraph._compressed_incidence_array.resize(_compressed_incidence_array.size());
      memcpy(hypergraph._compressed_incidence_array.data(), _compressed_incidence_array.data(),
             sizeof(uint8_t) * _compressed_incidence_array.size());
    }, [&] {
      hypergraph._community_ids = _community_ids;
    }, [&] {
      hypergraph._hypernode_weights = _hypernode_weights;
    }, [&] {
      hypergraph._hyperedge_weights = _hyperedge_weights;
    }, [&] {
      hypergraph.addFixedVertexSupport(_fixed_vertices.copy());
    });
    return hypergraph;
  }

  // ! Copy compressed hypergraph sequential
  CompressedHypergraph CompressedHypergraph::copy() const {
    CompressedHypergraph hypergraph;

    hypergraph._num_hypernodes = _num_hypernodes;
    hypergraph._num_removed_hypernodes = _num_removed_hypernodes;
    hypergraph._num_hyperedges = _num_hyperedges;
    hypergraph._num_removed_hyperedges = _num_removed_hyperedges;
    hypergraph._max_edge_size = _max_edge_size;
    hypergraph._num_pins = _num_pins;
    hypergraph._total_degree = _total_degree;
    hypergraph._total_weight = _total_weight;

  hypergraph._hypernode_offsets = _hypernode_offsets;
  hypergraph._hypernode_enabled = _hypernode_enabled;

    hypergraph._compressed_incident_nets.resize(_compressed_incident_nets.size());
    // FIX: copy bytes
    memcpy(hypergraph._compressed_incident_nets.data(), _compressed_incident_nets.data(),
           sizeof(uint8_t) * _compressed_incident_nets.size());

  hypergraph._hyperedge_offsets = _hyperedge_offsets;
  hypergraph._hyperedge_enabled = _hyperedge_enabled;

    hypergraph._compressed_incidence_array.resize(_compressed_incidence_array.size());
    // FIX: copy bytes
    memcpy(hypergraph._compressed_incidence_array.data(), _compressed_incidence_array.data(),
           sizeof(uint8_t) * _compressed_incidence_array.size());

  hypergraph._community_ids = _community_ids;
  hypergraph._hypernode_weights = _hypernode_weights;
  hypergraph._hyperedge_weights = _hyperedge_weights;
    hypergraph.addFixedVertexSupport(_fixed_vertices.copy());

    return hypergraph;
  }

  void CompressedHypergraph::memoryConsumption(utils::MemoryTreeNode* parent) const {
    ASSERT(parent);
  parent->addChild("Hypernode Offsets", sizeof(size_t) * _hypernode_offsets.capacity());
  parent->addChild("Hypernode Enabled (bits)", (_hypernode_enabled.capacity() + 7) / 8);
    parent->addChild("Incident Nets", sizeof(uint8_t) * _compressed_incident_nets.size());
  parent->addChild("Hyperedge Offsets", sizeof(size_t) * _hyperedge_offsets.capacity());
  parent->addChild("Hyperedge Enabled (bits)", (_hyperedge_enabled.capacity() + 7) / 8);
    parent->addChild("Incidence Array", sizeof(uint8_t) * _compressed_incidence_array.size());
    parent->addChild("Communities", sizeof(PartitionID) * _community_ids.capacity());
    if ( hasFixedVertices() ) {
      parent->addChild("Fixed Vertex Support", _fixed_vertices.size_in_bytes());
    }
  }

  // ! Computes the total node weight of the hypergraph
  void CompressedHypergraph::computeAndSetTotalNodeWeight(parallel_tag_t) {
    _total_weight = tbb::parallel_reduce(tbb::blocked_range<HypernodeID>(ID(0), _num_hypernodes), 0,
                                         [this](const tbb::blocked_range<HypernodeID>& range, HypernodeWeight init) {
                                           HypernodeWeight weight = init;
                                           for (HypernodeID hn = range.begin(); hn < range.end(); ++hn) {
                                             if (nodeIsEnabled(hn)) {
                                               weight += this->nodeWeight(hn);
                                             }
                                           }
                                           return weight;
                                         }, std::plus<>());
  }

} // namespace

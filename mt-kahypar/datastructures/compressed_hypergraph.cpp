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
  CompressedHypergraph CompressedHypergraph::contract(parallel::scalable_vector<HypernodeID>& /* communities */, bool /* deterministic*/) {
    throw UnsupportedOperationException(
      "Contraction not yet implemented for compressed hypergraph");
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
      hypergraph._hypernodes.resize(_hypernodes.size());
      memcpy(hypergraph._hypernodes.data(), _hypernodes.data(),
             sizeof(Hypernode) * _hypernodes.size());
    }, [&] {
      hypergraph._compressed_incident_nets.resize(_compressed_incident_nets.size());
      memcpy(hypergraph._compressed_incident_nets.data(), _compressed_incident_nets.data(),
             sizeof(HyperedgeID) * _compressed_incident_nets.size());
    }, [&] {
      hypergraph._hyperedges.resize(_hyperedges.size());
      memcpy(hypergraph._hyperedges.data(), _hyperedges.data(),
             sizeof(Hyperedge) * _hyperedges.size());
    }, [&] {
      hypergraph._compressed_incidence_array.resize(_compressed_incidence_array.size());
      memcpy(hypergraph._compressed_incidence_array.data(), _compressed_incidence_array.data(),
             sizeof(HypernodeID) * _compressed_incidence_array.size());
    }, [&] {
      hypergraph._community_ids = _community_ids;
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

    hypergraph._hypernodes.resize(_hypernodes.size());
    memcpy(hypergraph._hypernodes.data(), _hypernodes.data(),
           sizeof(Hypernode) * _hypernodes.size());
    hypergraph._compressed_incident_nets.resize(_compressed_incident_nets.size());
    memcpy(hypergraph._compressed_incident_nets.data(), _compressed_incident_nets.data(),
           sizeof(HyperedgeID) * _compressed_incident_nets.size());

    hypergraph._hyperedges.resize(_hyperedges.size());
    memcpy(hypergraph._hyperedges.data(), _hyperedges.data(),
           sizeof(Hyperedge) * _hyperedges.size());
    hypergraph._compressed_incidence_array.resize(_compressed_incidence_array.size());
    memcpy(hypergraph._compressed_incidence_array.data(), _compressed_incidence_array.data(),
           sizeof(HypernodeID) * _compressed_incidence_array.size());

    hypergraph._community_ids = _community_ids;
    hypergraph.addFixedVertexSupport(_fixed_vertices.copy());

    return hypergraph;
  }

  void CompressedHypergraph::memoryConsumption(utils::MemoryTreeNode* parent) const {
    ASSERT(parent);
    parent->addChild("Hypernodes", sizeof(Hypernode) * _hypernodes.size());
    parent->addChild("Incident Nets", sizeof(HyperedgeID) * _compressed_incident_nets.size());
    parent->addChild("Hyperedges", sizeof(Hyperedge) * _hyperedges.size());
    parent->addChild("Incidence Array", sizeof(HypernodeID) * _compressed_incidence_array.size());
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
                                               weight += this->_hypernodes[hn].weight();
                                             }
                                           }
                                           return weight;
                                         }, std::plus<>());
  }

} // namespace

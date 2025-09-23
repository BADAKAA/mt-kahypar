/*******************************************************************************
 * MIT License
 *
 * This file is part of Mt-KaHyPar.
 *
 * Copyright (C) 2019 Lars Gottesbüren
 * Copyright (C) 2019 Tobias Heuer
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

#pragma once

#include <vector>
#include <functional>
#include <cstdint>
#include <algorithm>
#include <cstring>
#include <unordered_map>
#include "include/mtkahypartypes.h"
#include "mt-kahypar/macros.h"
#include "mt-kahypar/datastructures/array.h"
#include "mt-kahypar/datastructures/hypergraph_common.h"
#include "mt-kahypar/datastructures/fixed_vertex_support.h"
#include "mt-kahypar/partition/context_enum_classes.h"
#include "mt-kahypar/utils/memory_tree.h"
#include "mt-kahypar/utils/range.h"
#include "mt-kahypar/utils/exception.h"
#include "mt-kahypar/parallel/stl/scalable_vector.h"

namespace mt_kahypar {
namespace ds {

// Forward declarations
class CompressedHypergraphFactory;
template <typename Hypergraph,
          typename ConnectivityInformation>
class PartitionedHypergraph;

// ####################### Compression Utility Functions #######################

inline void encode_varint(uint64_t value, std::vector<uint8_t> &out) {
  while (value >= 0x80) {
    out.push_back(static_cast<uint8_t>(value & 0x7F) | 0x80);
    value >>= 7;
  }
  out.push_back(static_cast<uint8_t>(value & 0x7F));
}

inline uint64_t decode_varint(const std::vector<uint8_t> &in, size_t &pos) {
  uint64_t result = 0;
  int shift = 0;
  while (pos < in.size()) {
    const uint8_t byte = in[pos++];
    result |= (static_cast<uint64_t>(byte & 0x7F) << shift);
    if ((byte & 0x80) == 0) break;
    shift += 7;
  }
  return result;
}

inline uint64_t decode_varint_bounded(const std::vector<uint8_t>& in, size_t& pos, const size_t limit) {
  uint64_t result = 0;
  int shift = 0;
  while (pos < limit) {
    const uint8_t byte = in[pos++];
    result |= (static_cast<uint64_t>(byte & 0x7F) << shift);
    if ((byte & 0x80) == 0) break;   // last byte
    shift += 7;
  }
  return result;
}

// Sort and deduplicate before gap encoding to guarantee a monotone sequence
inline void append_compressed_sequence(const std::vector<HypernodeID> &sequence, std::vector<uint8_t> &target) {
  if (sequence.empty()) return;
  std::vector<HypernodeID> sorted = sequence;
  std::sort(sorted.begin(), sorted.end());
  sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
  HypernodeID prev = 0;
  for (const HypernodeID node : sorted) {
    ASSERT(node >= prev);
    const HypernodeID gap = node - prev;
    encode_varint(gap, target);
    prev = node;
  }
}

inline std::vector<uint8_t> compress_sequence(const std::vector<HypernodeID> &sequence) {
  std::vector<uint8_t> encoded;
  append_compressed_sequence(sequence, encoded);
  return encoded;
}

class CompressedHypergraph {
    static constexpr bool enable_heavy_assert = false;
    static constexpr HyperedgeID HIGH_DEGREE_CONTRACTION_THRESHOLD = ID(500000);
    
    static_assert(std::is_unsigned<HypernodeID>::value, "Hypernode ID must be unsigned");
    static_assert(std::is_unsigned<HyperedgeID>::value, "Hyperedge ID must be unsigned");
    
    using UncontractionFunction = std::function<void(const HypernodeID, const HypernodeID, const HyperedgeID)>;

#define NOOP_BATCH_FUNC [] (const HypernodeID, const HypernodeID, const HyperedgeID) { }

    /**
     * Represents a hypernode with compressed incident nets storage
     */
    class Hypernode {
    public:
        using IDType = HypernodeID;

        Hypernode() :
            _begin(0),
            _compressed_size(0),
            _uncompressed_size(0),
            _weight(1),
            _valid(false) { }

        Hypernode(const bool valid) :
            _begin(0),
            _compressed_size(0),
            _uncompressed_size(0),
            _weight(1),
            _valid(valid) { }

        // Sentinel Constructor
        Hypernode(const size_t begin) :
            _begin(begin),
            _compressed_size(0),
            _uncompressed_size(0),
            _weight(1),
            _valid(false) { }

        bool isDisabled() const {
            return _valid == false;
        }

        void enable() {
            ASSERT(isDisabled());
            _valid = true;
        }

        void disable() {
            ASSERT(!isDisabled());
            _valid = false;
        }

        // Returns the index of the first element in compressed incident nets
        size_t firstEntry() const {
            return _begin;
        }

        // Sets the index of the first element in compressed incident nets
        void setFirstEntry(size_t begin) {
            ASSERT(!isDisabled());
            _begin = begin;
        }

        // Returns the index after the last element in compressed incident nets
        size_t firstInvalidEntry() const {
            return _begin + _compressed_size;
        }

        // Returns the compressed size in bytes
        size_t compressedSize() const {
            ASSERT(!isDisabled());
            return _compressed_size;
        }

        // Sets the compressed size in bytes
        void setCompressedSize(size_t size) {
            ASSERT(!isDisabled());
            _compressed_size = size;
        }

        // Returns the uncompressed size (number of incident nets)
        size_t size() const {
            ASSERT(!isDisabled());
            return _uncompressed_size;
        }

        // Sets the uncompressed size (number of incident nets)
        void setUncompressedSize(size_t size) {
            ASSERT(!isDisabled());
            _uncompressed_size = size;
        }

        HypernodeWeight weight() const {
            return _weight;
        }

        void setWeight(HypernodeWeight weight) {
            ASSERT(!isDisabled());
            _weight = weight;
        }

    private:
        size_t _begin;                  // Index in compressed incident nets array
        size_t _compressed_size;        // Size in compressed array (bytes)
        size_t _uncompressed_size;      // Number of incident nets when decompressed
        HypernodeWeight _weight;        // Hypernode weight
        bool _valid;                    // Flag indicating whether the element is active
    };

    /**
     * Represents a hyperedge with compressed pins storage
     */
    class Hyperedge {
    public:
        using IDType = HyperedgeID;

        Hyperedge() :
            _begin(0),
            _compressed_size(0),
            _uncompressed_size(0),
            _weight(1),
            _valid(false) { }

        // Sentinel Constructor
        Hyperedge(const size_t begin) :
            _begin(begin),
            _compressed_size(0),
            _uncompressed_size(0),
            _weight(1),
            _valid(false) { }

        void disable() {
            ASSERT(!isDisabled());
            _valid = false;
        }

        void enable() {
            ASSERT(isDisabled());
            _valid = true;
        }

        bool isDisabled() const {
            return _valid == false;
        }

        // Returns the index of the first element in compressed incidence array
        size_t firstEntry() const {
            return _begin;
        }

        // Sets the index of the first element in compressed incidence array
        void setFirstEntry(size_t begin) {
            ASSERT(!isDisabled());
            _begin = begin;
        }

        // Returns the index after the last element in compressed incidence array
        size_t firstInvalidEntry() const {
            return _begin + _compressed_size;
        }

        // Returns the compressed size in bytes
        size_t compressedSize() const {
            ASSERT(!isDisabled());
            return _compressed_size;
        }

        // Sets the compressed size in bytes
        void setCompressedSize(size_t size) {
            ASSERT(!isDisabled());
            _compressed_size = size;
        }

        // Returns the uncompressed size (number of pins)
        size_t size() const {
            ASSERT(!isDisabled());
            return _uncompressed_size;
        }

        // Sets the uncompressed size (number of pins)
        void setUncompressedSize(size_t size) {
            ASSERT(!isDisabled());
            _uncompressed_size = size;
        }

        HyperedgeWeight weight() const {
            ASSERT(!isDisabled());
            return _weight;
        }

        void setWeight(HyperedgeWeight weight) {
            ASSERT(!isDisabled());
            _weight = weight;
        }

        bool operator== (const Hyperedge& other) const {
            return _begin == other._begin && _compressed_size == other._compressed_size && 
                   _uncompressed_size == other._uncompressed_size && _weight == other._weight;
        }

        bool operator!= (const Hyperedge& other) const {
            return !(*this == other);
        }

    private:
        size_t _begin;                  // Index in compressed incidence array
        size_t _compressed_size;        // Length in compressed array
        size_t _uncompressed_size;      // Number of pins when decompressed
        HyperedgeWeight _weight;        // Hyperedge weight
        bool _valid;                    // Flag indicating whether the element is active
    };

    /**
     * Compressed Iterator for incident nets and pins
     * Decompresses data on-the-fly during iteration
     */
    template<typename ElementType>
    class CompressedIterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = ElementType;
        using reference = ElementType;
        using pointer = const ElementType*;
        using difference_type = std::ptrdiff_t;

        CompressedIterator(const std::vector<uint8_t>& compressed_data, 
                          size_t start_pos, size_t end_pos) :
            _compressed_data(compressed_data),
            _start_pos(start_pos),
            _end_pos(end_pos),
            _current_pos(start_pos),
            _current_value(0),
            _accumulated_value(0),
            _has_value(false) {
            if (_current_pos < _end_pos) {
                advance();
                _has_value = true;
            }
        }

        // End iterator
        CompressedIterator(const std::vector<uint8_t>& compressed_data, size_t end_pos) :
            _compressed_data(compressed_data),
            _start_pos(end_pos),
            _end_pos(end_pos),
            _current_pos(end_pos),
            _current_value(0),
            _accumulated_value(0),
            _has_value(false) {}

        ElementType operator*() const {
            ASSERT(_has_value);
            return _current_value;
        }

        CompressedIterator& operator++() {
            if (_has_value) advance();
            return *this;
        }

        CompressedIterator operator++(int) {
            CompressedIterator copy = *this;
            ++(*this);
            return copy;
        }

        // Compare iterator state by position/bounds to be robust
        bool operator!=(const CompressedIterator& rhs) const {
            // Both at end
            if (!_has_value && !rhs._has_value) return false;
            return _current_pos != rhs._current_pos || _end_pos != rhs._end_pos || _has_value != rhs._has_value;
        }

        bool operator==(const CompressedIterator& rhs) const {
            return !(*this != rhs);
        }

    private:
        void advance() {
            if (_current_pos >= _end_pos) {
                _has_value = false;
                return;
            };
            const uint64_t gap = decode_varint_bounded(_compressed_data, _current_pos, _end_pos);
            _accumulated_value += static_cast<ElementType>(gap);
            _current_value = _accumulated_value;
        }

        const std::vector<uint8_t>& _compressed_data;
        size_t _start_pos;
        size_t _end_pos;
        size_t _current_pos;
        ElementType _current_value;
        ElementType _accumulated_value;
        bool _has_value;
    };

    /**
     * Regular element iterator (for nodes/edges)
     */
    template<typename ElementType>
    class HypergraphElementIterator {
    public:
        using IDType = typename ElementType::IDType;
        using iterator_category = std::forward_iterator_tag;
        using value_type = IDType;
        using reference = IDType&;
        using pointer = const IDType*;
        using difference_type = std::ptrdiff_t;

        HypergraphElementIterator(const ElementType* start_element, IDType id, IDType max_id) :
            _id(id),
            _max_id(max_id),
            _element(start_element) {
            if (_id != _max_id && _element->isDisabled()) {
                operator++();
            }
        }

        IDType operator*() const {
            return _id;
        }

        HypergraphElementIterator& operator++() {
            ASSERT(_id < _max_id);
            do {
                ++_id;
                ++_element;
            } while (_id < _max_id && _element->isDisabled());
            return *this;
        }

        HypergraphElementIterator operator++(int) {
            HypergraphElementIterator copy = *this;
            operator++();
            return copy;
        }

        bool operator!=(const HypergraphElementIterator& rhs) {
            return _id != rhs._id;
        }

        bool operator==(const HypergraphElementIterator& rhs) {
            return _id == rhs._id;
        }

    private:
        IDType _id = 0;
        IDType _max_id = 0;
        const ElementType* _element = nullptr;
    };

    static_assert(std::is_trivially_copyable<Hypernode>::value, "Hypernode is not trivially copyable");
    static_assert(std::is_trivially_copyable<Hyperedge>::value, "Hyperedge is not trivially copyable");

    using CompressedIncidenceArray = std::vector<uint8_t>;    // Compressed pins storage
    using CompressedIncidentNets = std::vector<uint8_t>;      // Compressed incident nets storage

    // Contains buffers needed during contractions
    struct TmpContractionBuffer {
        explicit TmpContractionBuffer(const HypernodeID num_hypernodes,
                                    const HyperedgeID num_hyperedges,
                                    const HyperedgeID num_pins) {
            mapping.resize(num_hypernodes);
            tmp_hypernodes.resize(num_hypernodes);
            tmp_incident_nets.resize(num_pins);
            tmp_num_incident_nets.resize(num_hypernodes);
            hn_weights.resize(num_hypernodes);
            tmp_hyperedges.resize(num_hyperedges);
            tmp_incidence_array.resize(num_pins);
            he_sizes.resize(num_hyperedges);
            valid_hyperedges.resize(num_hyperedges);
        }

        std::vector<HypernodeID> mapping;
        std::vector<Hypernode> tmp_hypernodes;
        CompressedIncidentNets tmp_incident_nets;
        std::vector<HyperedgeID> tmp_num_incident_nets;
        std::vector<HypernodeWeight> hn_weights;
        std::vector<Hyperedge> tmp_hyperedges;
        CompressedIncidenceArray tmp_incidence_array;
        std::vector<HypernodeID> he_sizes;
        std::vector<bool> valid_hyperedges;
    };

public:
    static constexpr bool is_graph = false;
    static constexpr bool is_static_hypergraph = true;
    static constexpr bool is_partitioned = false;
    static constexpr size_t SIZE_OF_HYPERNODE = sizeof(Hypernode);
    static constexpr size_t SIZE_OF_HYPEREDGE = sizeof(Hyperedge);
    static constexpr mt_kahypar_hypergraph_type_t TYPE = COMPRESSED_HYPERGRAPH;

    // Factory
    using Factory = CompressedHypergraphFactory;

    // Iterator types
    using HypernodeIterator = HypergraphElementIterator<Hypernode>;
    using HyperedgeIterator = HypergraphElementIterator<Hyperedge>;
    // Pins of a hyperedge are HypernodeID
    using IncidenceIterator = CompressedIterator<HypernodeID>;
        // Incident nets of a hypernode are HyperedgeID
        // We wrap decoding with a filtering iterator to skip out-of-range IDs defensively.
        class IncidentNetsFilteringIterator {
        public:
                using iterator_category = std::forward_iterator_tag;
                using value_type = HyperedgeID;
                using reference = HyperedgeID;
                using pointer = const HyperedgeID*;
                using difference_type = std::ptrdiff_t;

                IncidentNetsFilteringIterator(const std::vector<uint8_t>& data,
                                                                            size_t start_pos,
                                                                            size_t end_pos,
                                                                            HyperedgeID max_edge_id)
                    : _data(data), _end(end_pos), _pos(start_pos),
                        _acc(0), _cur(0), _has(false), _max(max_edge_id) {
                    advance_to_valid();
                }

                // End iterator
                IncidentNetsFilteringIterator(const std::vector<uint8_t>& data,
                                                                            size_t end_pos)
                    : _data(data), _end(end_pos), _pos(end_pos),
                        _acc(0), _cur(0), _has(false), _max(0) { }

                HyperedgeID operator*() const { ASSERT(_has); return _cur; }

                IncidentNetsFilteringIterator& operator++() { advance_to_valid(); return *this; }

                IncidentNetsFilteringIterator operator++(int) { auto copy = *this; ++(*this); return copy; }

                bool operator!=(const IncidentNetsFilteringIterator& rhs) const {
                    if (!_has && !rhs._has) return false;
                    return _pos != rhs._pos || _end != rhs._end || _has != rhs._has;
                }

                bool operator==(const IncidentNetsFilteringIterator& rhs) const { return !(*this != rhs); }

        private:
                void advance_to_valid() {
                    _has = false;
                    while (_pos < _end) {
                        const uint64_t gap = decode_varint_bounded(_data, _pos, _end);
                        _acc += static_cast<HyperedgeID>(gap);
                        if (_acc < _max) { _cur = _acc; _has = true; return; }
                        // else: skip invalid ID and continue
                    }
                }

                const std::vector<uint8_t>& _data;
                size_t _end;
                size_t _pos;
                HyperedgeID _acc;
                HyperedgeID _cur;
                bool _has;
                HyperedgeID _max;
        };

        using IncidentNetsIterator = IncidentNetsFilteringIterator;

    struct ParallelHyperedge {
        HyperedgeID removed_hyperedge;
        HyperedgeID representative;
    };

    explicit CompressedHypergraph() :
        _num_hypernodes(0),
        _num_removed_hypernodes(0),
        _removed_degree_zero_hn_weight(0),
        _num_hyperedges(0),
        _num_removed_hyperedges(0),
        _max_edge_size(0),
        _num_pins(0),
        _total_degree(0),
        _total_weight(0),
        _hypernodes(),
        _compressed_incident_nets(),
        _hyperedges(),
        _compressed_incidence_array(),
        _community_ids(0),
        _fixed_vertices(),
        _tmp_contraction_buffer(nullptr) { }

    explicit CompressedHypergraph(const std::string& filename,
        bool remove_single_pin_hes = true);
        
    CompressedHypergraph(const CompressedHypergraph&) = delete;
    CompressedHypergraph& operator=(const CompressedHypergraph&) = delete;

    CompressedHypergraph(CompressedHypergraph&& other) :
        _num_hypernodes(other._num_hypernodes),
        _num_removed_hypernodes(other._num_removed_hypernodes),
        _removed_degree_zero_hn_weight(other._removed_degree_zero_hn_weight),
        _num_hyperedges(other._num_hyperedges),
        _num_removed_hyperedges(other._num_removed_hyperedges),
        _max_edge_size(other._max_edge_size),
        _num_pins(other._num_pins),
        _total_degree(other._total_degree),
        _total_weight(other._total_weight),
        _hypernodes(std::move(other._hypernodes)),
        _compressed_incident_nets(std::move(other._compressed_incident_nets)),
        _hyperedges(std::move(other._hyperedges)),
        _compressed_incidence_array(std::move(other._compressed_incidence_array)),
        _community_ids(std::move(other._community_ids)),
        _fixed_vertices(std::move(other._fixed_vertices)),
        _tmp_contraction_buffer(std::move(other._tmp_contraction_buffer)) {
        _fixed_vertices.setHypergraph(this);
        other._tmp_contraction_buffer = nullptr;
    }

    CompressedHypergraph& operator=(CompressedHypergraph&& other) {
        if (this != &other) {
            _num_hypernodes = other._num_hypernodes;
            _num_removed_hypernodes = other._num_removed_hypernodes;
            _removed_degree_zero_hn_weight = other._removed_degree_zero_hn_weight;
            _num_hyperedges = other._num_hyperedges;
            _num_removed_hyperedges = other._num_removed_hyperedges;
            _max_edge_size = other._max_edge_size;
            _num_pins = other._num_pins;
            _total_degree = other._total_degree;
            _total_weight = other._total_weight;
            _hypernodes = std::move(other._hypernodes);
            _compressed_incident_nets = std::move(other._compressed_incident_nets);
            _hyperedges = std::move(other._hyperedges);
            _compressed_incidence_array = std::move(other._compressed_incidence_array);
            _community_ids = std::move(other._community_ids);
            _fixed_vertices = std::move(other._fixed_vertices);
            _fixed_vertices.setHypergraph(this);
            _tmp_contraction_buffer = std::move(other._tmp_contraction_buffer);
            other._tmp_contraction_buffer = nullptr;
        }
        return *this;
    }

    ~CompressedHypergraph() {
        freeInternalData();
    }

    // ####################### General Hypergraph Stats #######################
    
    HypernodeID initialNumNodes() const {
        return _num_hypernodes;
    }

    HypernodeID numRemovedHypernodes() const {
        return _num_removed_hypernodes;
    }

    HypernodeWeight weightOfRemovedDegreeZeroVertices() const {
        return _removed_degree_zero_hn_weight;
    }

    HyperedgeID initialNumEdges() const {
        return _num_hyperedges;
    }

    HyperedgeID numRemovedHyperedges() const {
        return _num_removed_hyperedges;
    }

    void setNumRemovedHyperedges(const HyperedgeID num_removed_hyperedges) {
        _num_removed_hyperedges = num_removed_hyperedges;
    }

    HypernodeID initialNumPins() const {
        return _num_pins;
    }

    HypernodeID initialTotalVertexDegree() const {
        return _total_degree;
    }

    HypernodeWeight totalWeight() const {
        return _total_weight;
    }

    void computeAndSetTotalNodeWeight(parallel_tag_t);


    // ####################### Iterators #######################

    template<typename F>
    void doParallelForAllNodes(const F& f) const {
        // Sequential version (parallel logic omitted as requested)
        for (HypernodeID hn = 0; hn < _num_hypernodes; ++hn) {
            if (nodeIsEnabled(hn)) {
                f(hn);
            }
        }
    }

    template<typename F>
    void doParallelForAllEdges(const F& f) const {
        // Sequential version (parallel logic omitted as requested)
        for (HyperedgeID he = 0; he < _num_hyperedges; ++he) {
            if (edgeIsEnabled(he)) {
                f(he);
            }
        }
    }

    IteratorRange<HypernodeIterator> nodes() const {
        return IteratorRange<HypernodeIterator>(
            HypernodeIterator(_hypernodes.data(), ID(0), _num_hypernodes),
            HypernodeIterator(_hypernodes.data() + _num_hypernodes, _num_hypernodes, _num_hypernodes));
    }

    IteratorRange<HyperedgeIterator> edges() const {
        return IteratorRange<HyperedgeIterator>(
            HyperedgeIterator(_hyperedges.data(), ID(0), _num_hyperedges),
            HyperedgeIterator(_hyperedges.data() + _num_hyperedges, _num_hyperedges, _num_hyperedges));
    }

    IteratorRange<IncidentNetsIterator> incidentEdges(const HypernodeID u) const {
        ASSERT(!hypernode(u).isDisabled(), "Hypernode" << u << "is disabled");
        const Hypernode& hn = hypernode(u);
        return IteratorRange<IncidentNetsIterator>(
            IncidentNetsIterator(_compressed_incident_nets, hn.firstEntry(), hn.firstInvalidEntry(), _num_hyperedges),
            IncidentNetsIterator(_compressed_incident_nets, hn.firstInvalidEntry()));
    }

    IteratorRange<IncidenceIterator> pins(const HyperedgeID e) const {
        ASSERT(!hyperedge(e).isDisabled(), "Hyperedge" << e << "is disabled");
        const Hyperedge& he = hyperedge(e);
        return IteratorRange<IncidenceIterator>(
            IncidenceIterator(_compressed_incidence_array, he.firstEntry(), he.firstInvalidEntry()),
            IncidenceIterator(_compressed_incidence_array, he.firstInvalidEntry()));
    }

    // ####################### Hypernode Information #######################

    HypernodeWeight nodeWeight(const HypernodeID u) const {
        return hypernode(u).weight();
    }

    void setNodeWeight(const HypernodeID u, const HypernodeWeight weight) {
        ASSERT(!hypernode(u).isDisabled(), "Hypernode" << u << "is disabled");
        return hypernode(u).setWeight(weight);
    }

    HyperedgeID nodeDegree(const HypernodeID u) const {
        ASSERT(!hypernode(u).isDisabled(), "Hypernode" << u << "is disabled");
        return hypernode(u).size();
    }

    bool nodeIsEnabled(const HypernodeID u) const {
        return !hypernode(u).isDisabled();
    }

    void enableHypernode(const HypernodeID u) {
        hypernode(u).enable();
    }

    void disableHypernode(const HypernodeID u) {
        hypernode(u).disable();
    }

    void removeHypernode(const HypernodeID u) {
        hypernode(u).disable();
        ++_num_removed_hypernodes;
    }

    void removeDegreeZeroHypernode(const HypernodeID u) {
        ASSERT(nodeDegree(u) == 0);
        _removed_degree_zero_hn_weight += nodeWeight(u);
        removeHypernode(u);
    }

    void restoreDegreeZeroHypernode(const HypernodeID u) {
        hypernode(u).enable();
        ASSERT(nodeDegree(u) == 0);
        _removed_degree_zero_hn_weight -= nodeWeight(u);
    }

    // ####################### Hyperedge Information #######################

    HypernodeWeight edgeWeight(const HyperedgeID e) const {
        ASSERT(!hyperedge(e).isDisabled(), "Hyperedge" << e << "is disabled");
        return hyperedge(e).weight();
    }

    void setEdgeWeight(const HyperedgeID e, const HyperedgeWeight weight) {
        ASSERT(!hyperedge(e).isDisabled(), "Hyperedge" << e << "is disabled");
        return hyperedge(e).setWeight(weight);
    }

    HypernodeID edgeSize(const HyperedgeID e) const {
        ASSERT(!hyperedge(e).isDisabled(), "Hyperedge" << e << "is disabled");
        return hyperedge(e).size();
    }

    HypernodeID maxEdgeSize() const {
        return _max_edge_size;
    }

    bool edgeIsEnabled(const HyperedgeID e) const {
        return !hyperedge(e).isDisabled();
    }

    void enableHyperedge(const HyperedgeID e) {
        hyperedge(e).enable();
    }

    void disableHyperedge(const HyperedgeID e) {
        hyperedge(e).disable();
    }

    PartitionID communityID(const HypernodeID u) const {
        return _community_ids[u];
    }

    void setCommunityID(const HypernodeID u, const PartitionID community_id) {
        _community_ids[u] = community_id;
    }

    // ####################### Fixed Vertex Support #######################

    void addFixedVertexSupport(FixedVertexSupport<CompressedHypergraph>&& fixed_vertices) {
        _fixed_vertices = std::move(fixed_vertices);
        _fixed_vertices.setHypergraph(this);
    }

    bool hasFixedVertices() const {
        return _fixed_vertices.hasFixedVertices();
    }

    HypernodeWeight totalFixedVertexWeight() const {
        return _fixed_vertices.totalFixedVertexWeight();
    }

    HypernodeWeight fixedVertexBlockWeight(const PartitionID block) const {
        return _fixed_vertices.fixedVertexBlockWeight(block);
    }

    bool isFixed(const HypernodeID hn) const {
        return _fixed_vertices.isFixed(hn);
    }

    PartitionID fixedVertexBlock(const HypernodeID hn) const {
        return _fixed_vertices.fixedVertexBlock(hn);
    }

    void setMaxFixedVertexBlockWeight(const std::vector<HypernodeWeight>& max_block_weights) {
        _fixed_vertices.setMaxBlockWeight(max_block_weights);
    }

    const FixedVertexSupport<CompressedHypergraph>& fixedVertexSupport() const {
        return _fixed_vertices;
    }

    FixedVertexSupport<CompressedHypergraph> copyOfFixedVertexSupport() const {
        return _fixed_vertices.copy();
    }

    // ####################### Contract / Uncontract #######################

    // COMPRESSION_TODO
    CompressedHypergraph contract(parallel::scalable_vector<HypernodeID>& communities, bool deterministic = false);

    bool registerContraction(const HypernodeID, const HypernodeID) {
        throw UnsupportedOperationException(
            "registerContraction(u, v) is not supported in compressed hypergraph");
        return false;
    }

    size_t contract(const HypernodeID,
                    const HypernodeWeight max_node_weight = std::numeric_limits<HypernodeWeight>::max()) {
        unused(max_node_weight);
        throw UnsupportedOperationException(
        "contract(v, max_node_weight) is not supported in compressed hypergraph");
        return 0;
    }

    void uncontract(const Batch&,
                    const UncontractionFunction& case_one_func = NOOP_BATCH_FUNC,
                    const UncontractionFunction& case_two_func = NOOP_BATCH_FUNC) {
        unused(case_one_func);
        unused(case_two_func);
        throw UnsupportedOperationException(
        "uncontract(batch) is not supported in compressed hypergraph");
    }

    VersionedBatchVector createBatchUncontractionHierarchy(const size_t) {
        throw UnsupportedOperationException(
            "createBatchUncontractionHierarchy(batch_size) is not supported in compressed hypergraph");
        return {};
    }

    // ####################### Remove / Restore Hyperedges #######################

    void removeEdge(const HyperedgeID he) {
        ASSERT(edgeIsEnabled(he), "Hyperedge" << he << "is disabled");

        // Decode pins of the hyperedge
        const Hyperedge& edge = hyperedge(he);
        size_t he_pos = edge.firstEntry();
        const size_t he_end = edge.firstInvalidEntry();
        std::vector<HypernodeID> pins_of_he;
        pins_of_he.reserve(edge.size());
        {
            HypernodeID acc = 0;
            while (he_pos < he_end) {
                const uint64_t gap = decode_varint_bounded(_compressed_incidence_array, he_pos, he_end);
                acc += static_cast<HypernodeID>(gap);
                pins_of_he.push_back(acc);
            }
        }

        // For each pin, attempt in-place shrink; if any overflow, fallback to repack
        struct PendingUpdate { HypernodeID u; std::vector<uint8_t> bytes; size_t new_size; size_t begin; size_t cap; size_t new_uncompressed; };
        std::vector<PendingUpdate> pending; pending.reserve(pins_of_he.size());
        bool needs_repack = false;
        for (const HypernodeID u : pins_of_he) {
            if (!nodeIsEnabled(u)) continue;
            Hypernode& hn = hypernode(u);
            const size_t begin = hn.firstEntry();
            const size_t end = hn.firstInvalidEntry();
            // Decode current incidents
            std::vector<HyperedgeID> inc; inc.reserve(hn.size());
            size_t cur = begin; HyperedgeID acc = 0;
            while (cur < end) { const uint64_t gap = decode_varint_bounded(_compressed_incident_nets, cur, end); acc += static_cast<HyperedgeID>(gap); if (acc != he && acc < _num_hyperedges) inc.push_back(acc); }
            // Re-encode
            std::vector<uint8_t> bytes;
            if (!inc.empty()) {
                std::sort(inc.begin(), inc.end()); inc.erase(std::unique(inc.begin(), inc.end()), inc.end());
                HyperedgeID prev = 0; for (HyperedgeID x : inc) { ASSERT(x >= prev); encode_varint(static_cast<uint64_t>(x - prev), bytes); prev = x; }
            }
            const size_t new_size = bytes.size();
            // Capacity up to next enabled node
            HypernodeID v = u + 1; while (v < _num_hypernodes && !nodeIsEnabled(v)) ++v;
            size_t next_begin = (v < _num_hypernodes) ? hypernode(v).firstEntry() : _compressed_incident_nets.size();
            const size_t cap = next_begin - begin;
            if (new_size > cap) needs_repack = true;
            pending.push_back(PendingUpdate{u, std::move(bytes), new_size, begin, cap, inc.size()});
        }

        if (!needs_repack) {
            for (auto& up : pending) {
                Hypernode& hn = hypernode(up.u);
                if (up.new_size > 0) {
                    std::copy(up.bytes.begin(), up.bytes.end(), _compressed_incident_nets.begin() + up.begin);
                }
                hn.setCompressedSize(up.new_size);
                hn.setUncompressedSize(up.new_uncompressed);
            }
        } else {
            // Repack only if absolutely necessary (rare due to varint edge cases)
            std::unordered_map<HypernodeID, std::vector<uint8_t>> overrides;
            overrides.reserve(pending.size());
            for (const auto& up : pending) { overrides.emplace(up.u, up.bytes); hypernode(up.u).setUncompressedSize(up.new_uncompressed); }
            // Snapshot current begins/ends
            std::vector<size_t> old_begin(_num_hypernodes), old_end(_num_hypernodes);
            for (HypernodeID u = 0; u < _num_hypernodes; ++u) { const Hypernode& hn = hypernode(u); old_begin[u] = hn.firstEntry(); old_end[u] = hn.firstInvalidEntry(); }
            auto encode_from_snapshot = [&](HypernodeID u) {
                std::vector<HyperedgeID> inc; size_t cur = old_begin[u]; const size_t end = old_end[u]; HyperedgeID acc = 0;
                while (cur < end) { const uint64_t g = decode_varint_bounded(_compressed_incident_nets, cur, end); acc += static_cast<HyperedgeID>(g); if (acc < _num_hyperedges) inc.push_back(acc); }
                std::vector<uint8_t> out; if (!inc.empty()) { std::sort(inc.begin(), inc.end()); inc.erase(std::unique(inc.begin(), inc.end()), inc.end()); HyperedgeID prev = 0; for (HyperedgeID x : inc) { ASSERT(x >= prev); encode_varint(static_cast<uint64_t>(x - prev), out); prev = x; } }
                return out;
            };
            CompressedIncidentNets new_incident_nets; new_incident_nets.reserve(_compressed_incident_nets.size());
            size_t write_pos = 0;
            for (HypernodeID u = 0; u < _num_hypernodes; ++u) {
                Hypernode& hn = hypernode(u);
                if (!nodeIsEnabled(u)) continue;
                hn.setFirstEntry(write_pos);
                std::vector<uint8_t> bytes;
                auto it = overrides.find(u);
                if (it != overrides.end()) { bytes = it->second; } else { bytes = encode_from_snapshot(u); }
                new_incident_nets.insert(new_incident_nets.end(), bytes.begin(), bytes.end());
                hn.setCompressedSize(bytes.size());
                write_pos += bytes.size();
            }
            _compressed_incident_nets.swap(new_incident_nets);
        }

        ++_num_removed_hyperedges;
        disableHyperedge(he);
    }

    void removeLargeEdge(const HyperedgeID he) {
        ASSERT(edgeIsEnabled(he), "Hyperedge" << he << "is disabled");

        // Decode pins
        const Hyperedge& edge = hyperedge(he);
        size_t he_pos = edge.firstEntry();
        const size_t he_end = edge.firstInvalidEntry();
        std::vector<HypernodeID> pins_of_he;
        pins_of_he.reserve(edge.size());
        {
            HypernodeID acc = 0;
            while (he_pos < he_end) {
                const uint64_t gap = decode_varint_bounded(_compressed_incidence_array, he_pos, he_end);
                acc += static_cast<HypernodeID>(gap);
                pins_of_he.push_back(acc);
            }
        }

        // Attempt in-place shrink for each affected node; fallback to repack if any overflow
        struct PendingUpdateLE { HypernodeID u; std::vector<uint8_t> bytes; size_t new_size; size_t begin; size_t cap; size_t new_uncompressed; };
        std::vector<PendingUpdateLE> updates; updates.reserve(pins_of_he.size());
        bool needs_repack = false;
        for (const HypernodeID u : pins_of_he) {
            if (!nodeIsEnabled(u)) continue;
            Hypernode& hn = hypernode(u);
            const size_t begin = hn.firstEntry();
            const size_t end = hn.firstInvalidEntry();
            // Decode incidents
            std::vector<HyperedgeID> inc; inc.reserve(hn.size());
            size_t cur = begin; HyperedgeID acc = 0;
            while (cur < end) { const uint64_t gap = decode_varint_bounded(_compressed_incident_nets, cur, end); acc += static_cast<HyperedgeID>(gap); if (acc != he && acc < _num_hyperedges) inc.push_back(acc); }
            std::vector<uint8_t> bytes;
            if (!inc.empty()) { std::sort(inc.begin(), inc.end()); inc.erase(std::unique(inc.begin(), inc.end()), inc.end()); HyperedgeID prev = 0; for (HyperedgeID x : inc) { ASSERT(x >= prev); encode_varint(static_cast<uint64_t>(x - prev), bytes); prev = x; } }
            const size_t new_size = bytes.size();
            // Capacity to next enabled node
            HypernodeID v = u + 1; while (v < _num_hypernodes && !nodeIsEnabled(v)) ++v;
            const size_t next_begin = (v < _num_hypernodes) ? hypernode(v).firstEntry() : _compressed_incident_nets.size();
            const size_t cap = next_begin - begin;
            if (new_size > cap) needs_repack = true;
            updates.push_back(PendingUpdateLE{u, std::move(bytes), new_size, begin, cap, inc.size()});
        }
        if (!needs_repack) {
            for (auto& up : updates) {
                Hypernode& hn = hypernode(up.u);
                if (up.new_size > 0) {
                    std::copy(up.bytes.begin(), up.bytes.end(), _compressed_incident_nets.begin() + up.begin);
                }
                hn.setCompressedSize(up.new_size);
                hn.setUncompressedSize(up.new_uncompressed);
            }
        } else {
            // Repack fallback
            std::unordered_map<HypernodeID, std::vector<uint8_t>> overrides;
            overrides.reserve(updates.size());
            for (const auto& up : updates) { overrides.emplace(up.u, up.bytes); hypernode(up.u).setUncompressedSize(up.new_uncompressed); }
            std::vector<size_t> old_begin(_num_hypernodes), old_end(_num_hypernodes);
            for (HypernodeID u = 0; u < _num_hypernodes; ++u) { const Hypernode& hn = hypernode(u); old_begin[u] = hn.firstEntry(); old_end[u] = hn.firstInvalidEntry(); }
            auto encode_from_snapshot = [&](HypernodeID u) {
                std::vector<HyperedgeID> inc; size_t cur = old_begin[u]; const size_t end = old_end[u]; HyperedgeID acc = 0;
                while (cur < end) { const uint64_t g = decode_varint_bounded(_compressed_incident_nets, cur, end); acc += static_cast<HyperedgeID>(g); if (acc < _num_hyperedges) inc.push_back(acc); }
                std::vector<uint8_t> out; if (!inc.empty()) { std::sort(inc.begin(), inc.end()); inc.erase(std::unique(inc.begin(), inc.end()), inc.end()); HyperedgeID prev = 0; for (HyperedgeID x : inc) { ASSERT(x >= prev); encode_varint(static_cast<uint64_t>(x - prev), out); prev = x; } }
                return out;
            };
            CompressedIncidentNets new_incident_nets; new_incident_nets.reserve(_compressed_incident_nets.size());
            size_t write_pos = 0;
            for (HypernodeID u = 0; u < _num_hypernodes; ++u) {
                Hypernode& hn = hypernode(u);
                if (!nodeIsEnabled(u)) continue;
                hn.setFirstEntry(write_pos);
                std::vector<uint8_t> bytes;
                auto it = overrides.find(u);
                if (it != overrides.end()) { bytes = it->second; } else { bytes = encode_from_snapshot(u); }
                new_incident_nets.insert(new_incident_nets.end(), bytes.begin(), bytes.end());
                hn.setCompressedSize(bytes.size());
                write_pos += bytes.size();
            }
            _compressed_incident_nets.swap(new_incident_nets);
        }

        // Mirror static semantics: do not change removed count
        disableHyperedge(he);
    }

    void restoreLargeEdge(const HyperedgeID& he) {
        ASSERT(!edgeIsEnabled(he), "Hyperedge" << he << "is enabled");

        // Decode pins of the hyperedge
        const Hyperedge& edge = hyperedge(he);
        size_t he_pos = edge.firstEntry();
        const size_t he_end = edge.firstInvalidEntry();
        std::vector<HypernodeID> pins_of_he;
        pins_of_he.reserve(edge.size());
        {
            HypernodeID acc = 0;
            while (he_pos < he_end) {
                const uint64_t gap = decode_varint_bounded(_compressed_incidence_array, he_pos, he_end);
                acc += static_cast<HypernodeID>(gap);
                pins_of_he.push_back(acc);
            }
        }

        // First pass: compute new bytes per affected node and check if in-place growth fits
    struct PendingUpdate { HypernodeID u; std::vector<uint8_t> bytes; size_t new_size; size_t begin; size_t old_size; size_t cap; size_t new_uncompressed; };
        std::vector<PendingUpdate> pending; pending.reserve(pins_of_he.size());
        bool needs_repack = false;
        for (const HypernodeID u : pins_of_he) {
            if (!nodeIsEnabled(u)) continue;
            Hypernode& hn = hypernode(u);
            const size_t begin = hn.firstEntry();
            const size_t old_size = hn.compressedSize();
            const size_t end = begin + old_size;
            // Decode existing incidents
            std::vector<HyperedgeID> inc; inc.reserve(hn.size() + 1);
            size_t cur = begin; HyperedgeID acc = 0;
            while (cur < end) { const uint64_t g = decode_varint_bounded(_compressed_incident_nets, cur, end); acc += static_cast<HyperedgeID>(g); if (acc < _num_hyperedges) inc.push_back(acc); }
            // Insert he and encode
            inc.push_back(he);
            std::sort(inc.begin(), inc.end()); inc.erase(std::unique(inc.begin(), inc.end()), inc.end());
            std::vector<uint8_t> bytes; bytes.reserve(old_size + 4);
            HyperedgeID prev = 0; for (HyperedgeID x : inc) { ASSERT(x >= prev); encode_varint(static_cast<uint64_t>(x - prev), bytes); prev = x; }
            const size_t new_size = bytes.size();
            const size_t new_uncompressed = inc.size();
            // Compute available capacity up to next enabled node begin (or vector end)
            HypernodeID v = u + 1; while (v < _num_hypernodes && !nodeIsEnabled(v)) ++v;
            size_t next_begin = (v < _num_hypernodes) ? hypernode(v).firstEntry() : _compressed_incident_nets.size();
            size_t cap = next_begin - begin; // bytes we can occupy without moving neighbors
            if (new_size > cap) needs_repack = true;
            pending.push_back(PendingUpdate{u, std::move(bytes), new_size, begin, old_size, cap, new_uncompressed});
        }

        if (!needs_repack) {
            // In-place apply all updates (may extend vector for last node)
            for (auto& up : pending) {
                Hypernode& hn = hypernode(up.u);
                size_t needed_end = up.begin + up.new_size;
                if (needed_end > _compressed_incident_nets.size()) {
                    _compressed_incident_nets.resize(needed_end);
                }
                if (up.new_size > 0) {
                    std::copy(up.bytes.begin(), up.bytes.end(), _compressed_incident_nets.begin() + up.begin);
                }
                hn.setCompressedSize(up.new_size);
                hn.setUncompressedSize(up.new_uncompressed);
            }
        } else {
            // Fallback: single repack using prepared bytes for affected nodes
            std::unordered_map<HypernodeID, std::vector<uint8_t>> overrides;
            overrides.reserve(pending.size());
            for (const auto& up : pending) { overrides.emplace(up.u, up.bytes); }
            // Snapshot current begins/ends
            std::vector<size_t> old_begin(_num_hypernodes), old_end(_num_hypernodes);
            for (HypernodeID u = 0; u < _num_hypernodes; ++u) {
                const Hypernode& hn = hypernode(u);
                old_begin[u] = hn.firstEntry();
                old_end[u] = hn.firstInvalidEntry();
            }
            // Helper to decode an untouched node from snapshot and encode again
            auto encode_from_snapshot = [&](HypernodeID u) {
                std::vector<HyperedgeID> inc; size_t cur = old_begin[u]; const size_t end = old_end[u]; HyperedgeID acc = 0;
                while (cur < end) { const uint64_t g = decode_varint_bounded(_compressed_incident_nets, cur, end); acc += static_cast<HyperedgeID>(g); if (acc < _num_hyperedges) inc.push_back(acc); }
                std::vector<uint8_t> out; if (!inc.empty()) { std::sort(inc.begin(), inc.end()); inc.erase(std::unique(inc.begin(), inc.end()), inc.end()); HyperedgeID prev = 0; for (HyperedgeID x : inc) { ASSERT(x >= prev); encode_varint(static_cast<uint64_t>(x - prev), out); prev = x; } }
                return out;
            };
            CompressedIncidentNets new_incident_nets; new_incident_nets.reserve(_compressed_incident_nets.size() + 16);
            size_t write_pos = 0;
            for (HypernodeID u = 0; u < _num_hypernodes; ++u) {
                Hypernode& hn = hypernode(u);
                if (!nodeIsEnabled(u)) continue;
                hn.setFirstEntry(write_pos);
                std::vector<uint8_t> bytes;
                auto it = overrides.find(u);
                if (it != overrides.end()) {
                    bytes = it->second;
                } else {
                    bytes = encode_from_snapshot(u);
                }
                new_incident_nets.insert(new_incident_nets.end(), bytes.begin(), bytes.end());
                hn.setCompressedSize(bytes.size());
                write_pos += bytes.size();
            }
            _compressed_incident_nets.swap(new_incident_nets);
            // Update uncompressed sizes precisely for affected nodes in fallback path
            for (auto& up : pending) { hypernode(up.u).setUncompressedSize(up.new_uncompressed); }
        }

        // Re-enable the edge
        enableHyperedge(he);
    }

    parallel::scalable_vector<ParallelHyperedge> removeSinglePinAndParallelHyperedges() {
        throw UnsupportedOperationException(
            "removeSinglePinAndParallelHyperedges() is not supported in compressed hypergraph");
        return { };
    }

    void restoreSinglePinAndParallelNets(const parallel::scalable_vector<ParallelHyperedge>&) {
        throw UnsupportedOperationException(
            "restoreSinglePinAndParallelNets(hes_to_restore) is not supported in compressed hypergraph");
    }

    // ####################### Initialization / Reset Functions #######################


    // ! Reset internal community information
    void copyCommunityIDs(const parallel::scalable_vector<PartitionID>& community_ids) {
        ASSERT(community_ids.size() == UI64(_num_hypernodes));
        doParallelForAllNodes([&](const HypernodeID& hn) {
        _community_ids[hn] = community_ids[hn];
        });
    }

    void setCommunityIDs(ds::Clustering&& communities) {
        ASSERT(communities.size() == initialNumNodes());
        _community_ids = std::move(communities);
    }

    // ! Copy compressed hypergraph in parallel
    CompressedHypergraph copy(parallel_tag_t) const;

    // ! Copy compressed hypergraph sequential
    CompressedHypergraph copy() const;

    void reset() { }

    void freeInternalData() {
        // Delete any temporary contraction buffer
        freeTmpContractionBuffer();
        // Release main containers
        _hypernodes.clear();
        _hypernodes.shrink_to_fit();
        _compressed_incident_nets.clear();
        _compressed_incident_nets.shrink_to_fit();
        _hyperedges.clear();
        _hyperedges.shrink_to_fit();
        _compressed_incidence_array.clear();
        _compressed_incidence_array.shrink_to_fit();
        _community_ids.clear();
        _community_ids.shrink_to_fit();
        _fixed_vertices = FixedVertexSupport<CompressedHypergraph>();
        // Reset stats
        _num_hypernodes = 0;
        _num_removed_hypernodes = 0;
        _removed_degree_zero_hn_weight = 0;
        _num_hyperedges = 0;
        _num_removed_hyperedges = 0;
        _max_edge_size = 0;
        _num_pins = 0;
        _total_degree = 0;
        _total_weight = 0;
    }

    void freeTmpContractionBuffer() {
        if (_tmp_contraction_buffer) {
            delete(_tmp_contraction_buffer);
            _tmp_contraction_buffer = nullptr;
        }
    }

    void memoryConsumption(utils::MemoryTreeNode* parent) const;

    size_t memoryConsumptionKB() const {
        size_t total = 0;
        total += sizeof(Hypernode) * _hypernodes.size();
        total += sizeof(uint8_t) * _compressed_incident_nets.size();
        total += sizeof(Hyperedge) * _hyperedges.size();
        total += sizeof(uint8_t) * _compressed_incidence_array.size();
        total += sizeof(PartitionID) * _community_ids.capacity();
        if (_fixed_vertices.hasFixedVertices()) {
        total += _fixed_vertices.size_in_bytes();
        }
        return total / 1024;
    }

    bool verifyIncidenceArrayAndIncidentNets() {
        throw UnsupportedOperationException(
            "verifyIncidenceArrayAndIncidentNets() not supported in compressed hypergraph");
        return false;
    }

private:
    friend class CompressedHypergraphFactory;
    template <typename Hypergraph,
          typename ConnectivityInformation>
    friend class PartitionedHypergraph;
    template<typename Hypergraph>
    friend class CommunitySupport;

    // ####################### Hypernode Information #######################

    MT_KAHYPAR_ATTRIBUTE_ALWAYS_INLINE const Hypernode& hypernode(const HypernodeID u) const {
        ASSERT(u < _num_hypernodes, "Hypernode" << u << " does not exist");
        return _hypernodes[u];
    }

    MT_KAHYPAR_ATTRIBUTE_ALWAYS_INLINE Hypernode& hypernode(const HypernodeID u) {
        return const_cast<Hypernode&>(static_cast<const CompressedHypergraph&>(*this).hypernode(u));
    }

   MT_KAHYPAR_ATTRIBUTE_ALWAYS_INLINE IteratorRange<IncidentNetsIterator>
   incident_nets_of(const HypernodeID u, const size_t pos = 0) const {
       ASSERT(!hypernode(u).isDisabled(), "Hypernode" << u << "is disabled");
       if (pos != 0) {
           throw UnsupportedOperationException(
               "incident_nets_of(u, pos) with pos > 0 is not supported in compressed hypergraph");
        }
       const Hypernode& hn = hypernode(u);
       return IteratorRange<IncidentNetsIterator>(
           IncidentNetsIterator(_compressed_incident_nets, hn.firstEntry(), hn.firstInvalidEntry(), _num_hyperedges),
           IncidentNetsIterator(_compressed_incident_nets, hn.firstInvalidEntry()));
   }

    // ####################### Hyperedge Information #######################

    MT_KAHYPAR_ATTRIBUTE_ALWAYS_INLINE const Hyperedge& hyperedge(const HyperedgeID e) const {
        ASSERT(e < _num_hyperedges, "Hyperedge " << e << " does not exist");
        return _hyperedges[e];
    }

    MT_KAHYPAR_ATTRIBUTE_ALWAYS_INLINE Hyperedge& hyperedge(const HyperedgeID e) {
        return const_cast<Hyperedge&>(static_cast<const CompressedHypergraph&>(*this).hyperedge(e));
    }

    MT_KAHYPAR_ATTRIBUTE_ALWAYS_INLINE HypernodeID pinAt(const HyperedgeID e, const size_t local_pos) const {
        ASSERT(!hyperedge(e).isDisabled(), "Hyperedge" << e << "is disabled");
        ASSERT(local_pos < edgeSize(e));
        // Decode varints up to local_pos
        const Hyperedge& he = hyperedge(e);
        size_t cur = he.firstEntry();
        const size_t end = he.firstInvalidEntry();
        HypernodeID acc = 0;
        for (size_t i = 0; i <= local_pos; ++i) {
            ASSERT(cur < end);
            const uint64_t gap = decode_varint_bounded(_compressed_incidence_array, cur, end);
            acc += static_cast<HypernodeID>(gap);
        }
        return acc;
    }

    void allocateTmpContractionBuffer() {
        if (!_tmp_contraction_buffer) {
            _tmp_contraction_buffer = new TmpContractionBuffer(
                _num_hypernodes, _num_hyperedges, _num_pins);
        }
    }

    // Member variables
    HypernodeID _num_hypernodes;
    HypernodeID _num_removed_hypernodes;
    HypernodeWeight _removed_degree_zero_hn_weight;
    HyperedgeID _num_hyperedges;
    HyperedgeID _num_removed_hyperedges;
    HypernodeID _max_edge_size;
    HypernodeID _num_pins;
    HypernodeID _total_degree;
    HypernodeWeight _total_weight;

    // Hypergraph structure with compression
    std::vector<Hypernode> _hypernodes;
    CompressedIncidentNets _compressed_incident_nets;    // Compressed incident nets storage
    std::vector<Hyperedge> _hyperedges;
    CompressedIncidenceArray _compressed_incidence_array;  // Compressed pins storage

    // Communities and fixed vertices (uncompressed as requested)
    ds::Clustering _community_ids;
    FixedVertexSupport<CompressedHypergraph> _fixed_vertices;

    // Data reused throughout multilevel hierarchy
    TmpContractionBuffer* _tmp_contraction_buffer = nullptr;
};

} // namespace ds
} // namespace mt_kahypar
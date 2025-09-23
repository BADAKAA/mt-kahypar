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
#include "mt-kahypar/datastructures/bit_vector.h"

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

// Encode an unsigned LEB128 (varint) of a fixed total length (>= minimal).
// If fixed_len equals the minimal length, this produces the canonical encoding.
// If fixed_len is larger, we extend the encoding by appending zero-groups at the
// most-significant end (keeping group alignment) which still decodes to the same value.
// This allows in-place updates without shifting following bytes.
inline void encode_varint_fixed_len(uint64_t value, size_t fixed_len, std::vector<uint8_t>& out_bytes) {
    out_bytes.clear();
    // First, generate minimal bytes
    std::vector<uint8_t> tmp;
    encode_varint(value, tmp);
    if (fixed_len < tmp.size()) {
        throw UnsupportedOperationException("encode_varint_fixed_len: fixed_len too small for value");
    }
    if (fixed_len == tmp.size()) {
        out_bytes = std::move(tmp);
        return;
    }
    // Extend by converting the last byte to continuation and appending zero groups
    // until we reach fixed_len - 1, then a final stop byte 0x00.
    // Example: value=1 (0x01) to len=2 -> [0x81, 0x00]
    //          value=300 (0xAC 0x02) to len=4 -> [0xAC|0x80, 0x02|0x80, 0x80, 0x00]
    std::vector<uint8_t> extended;
    // Copy all but last minimal byte as-is
    if (!tmp.empty()) {
        for (size_t i = 0; i + 1 < tmp.size(); ++i) {
            // ensure continuation bit set for all non-final groups
            extended.push_back(tmp[i] | 0x80u);
        }
        // Last minimal byte becomes continuation as well (unless minimal was already multiple bytes where its MSB may already be 0)
        extended.push_back(tmp.back() | 0x80u);
    } else {
        // value == 0 minimal representation empty should not happen (encode_varint always emits at least one byte)
        extended.push_back(0x80u);
    }
    // Append (fixed_len - minimal_len - 1) zero continuation groups
    const size_t extra_groups = fixed_len - tmp.size() - 1;
    for (size_t i = 0; i < extra_groups; ++i) extended.push_back(0x80u);
    // Final stop byte with zero payload
    extended.push_back(0x00u);
    out_bytes = std::move(extended);
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

    // Proxy for a hypernode backed by parent offsets/enabled arrays
    class Hypernode {
    public:
        using IDType = HypernodeID;
        Hypernode() : _hg(nullptr), _id(0) {}
        Hypernode(const CompressedHypergraph* hg, HypernodeID id) : _hg(hg), _id(id) {}

    bool isDisabled() const { return !_hg || !_hg->_hypernode_enabled[_id]; }
    void enable() const { ASSERT(_hg); const_cast<CompressedHypergraph*>(_hg)->_hypernode_enabled[_id] = true; }
    void disable() const { ASSERT(_hg); const_cast<CompressedHypergraph*>(_hg)->_hypernode_enabled[_id] = false; }

        size_t firstEntry() const { return _hg->_hypernode_offsets[_id]; }
        void setFirstEntry(const size_t begin) const { ASSERT(!isDisabled()); const_cast<CompressedHypergraph*>(_hg)->_hypernode_offsets[_id] = begin; }

        size_t firstInvalidEntry() const { return _hg->hypernode_firstInvalidEntry(_id); }
        HyperedgeID degree() const { return _hg->nodeDegree(_id); }
        HypernodeWeight weight() const { return _hg->nodeWeight(_id); }
        void setWeight(HypernodeWeight w) const { const_cast<CompressedHypergraph*>(_hg)->setNodeWeight(_id, w); }
        HypernodeID id() const { return _id; }

    private:
        const CompressedHypergraph* _hg;
        HypernodeID _id;
    };

    // Proxy for a hyperedge
    class Hyperedge {
    public:
        using IDType = HyperedgeID;
        Hyperedge() : _hg(nullptr), _id(0) {}
        Hyperedge(const CompressedHypergraph* hg, HyperedgeID id) : _hg(hg), _id(id) {}

    bool isDisabled() const { return !_hg || !_hg->_hyperedge_enabled[_id]; }
    void enable() const { ASSERT(_hg); const_cast<CompressedHypergraph*>(_hg)->_hyperedge_enabled[_id] = true; }
    void disable() const { ASSERT(_hg); const_cast<CompressedHypergraph*>(_hg)->_hyperedge_enabled[_id] = false; }

        size_t firstEntry() const { return _hg->_hyperedge_offsets[_id]; }
        void setFirstEntry(const size_t begin) const { ASSERT(!isDisabled()); const_cast<CompressedHypergraph*>(_hg)->_hyperedge_offsets[_id] = begin; }

        size_t firstInvalidEntry() const { return _hg->hyperedge_firstInvalidEntry(_id); }
        HypernodeID size() const { return _hg->edgeSize(_id); }
        HyperedgeWeight weight() const { return _hg->edgeWeight(_id); }
        void setWeight(HyperedgeWeight w) const { const_cast<CompressedHypergraph*>(_hg)->setEdgeWeight(_id, w); }
        HyperedgeID id() const { return _id; }

        bool operator==(const Hyperedge& other) const { return _hg == other._hg && _id == other._id; }
        bool operator!=(const Hyperedge& other) const { return !(*this == other); }

    private:
        const CompressedHypergraph* _hg;
        HyperedgeID _id;
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

    // Iterators over IDs using enabled flags
    template <typename ID>
    class IDIterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = ID;
        using reference = ID;
        using pointer = void;
        using difference_type = std::ptrdiff_t;

        IDIterator(const CompressedHypergraph* hg, ID id, ID max, bool edge) :
            _hg(hg), _id(id), _max(max), _edge(edge) {
            if (_id != _max && !isEnabled(_id)) ++(*this);
        }

        ID operator*() const { return _id; }

        IDIterator& operator++() {
            do { ++_id; } while (_id < _max && !isEnabled(_id));
            return *this;
        }

        IDIterator operator++(int) { auto c = *this; ++(*this); return c; }
        bool operator!=(const IDIterator& rhs) const { return _id != rhs._id; }
        bool operator==(const IDIterator& rhs) const { return _id == rhs._id; }

    private:
        bool isEnabled(ID x) const {
            return _edge ? _hg->_hyperedge_enabled[x]
                         : _hg->_hypernode_enabled[x];
        }
        const CompressedHypergraph* _hg;
        ID _id;
        ID _max;
        bool _edge;
    };

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

    // Iterator types (ID-based over enabled flags)
    // Note: legacy element iterator from other hypergraph variants is not used here
        // Pins of a hyperedge are HypernodeID (degree-bounded, skip zero-gap placeholders)
        class PinsIterator {
        public:
                using iterator_category = std::forward_iterator_tag;
                using value_type = HypernodeID;
                using reference = HypernodeID;
                using pointer = const HypernodeID*;
                using difference_type = std::ptrdiff_t;

                PinsIterator(const std::vector<uint8_t>& data,
                                         size_t start_pos,
                                         size_t end_pos,
                                         size_t remaining)
                        : _data(data), _end(end_pos), _pos(start_pos), _acc(0), _cur(0), _has(false), _left(remaining), _emitted(0) {
                    advance_to_valid();
                }

                PinsIterator(const std::vector<uint8_t>& data, size_t end_pos)
                        : _data(data), _end(end_pos), _pos(end_pos), _acc(0), _cur(0), _has(false), _left(0), _emitted(0) {}

                HypernodeID operator*() const { ASSERT(_has); return _cur; }
                PinsIterator& operator++() { if (_has) advance_to_valid(); return *this; }
                PinsIterator operator++(int) { auto c = *this; ++(*this); return c; }
                bool operator!=(const PinsIterator& rhs) const { if (!_has && !rhs._has) return false; return _pos != rhs._pos || _end != rhs._end || _has != rhs._has || _left != rhs._left; }
                bool operator==(const PinsIterator& rhs) const { return !(*this != rhs); }

        private:
                void advance_to_valid() {
                    _has = false;
                    while (_pos < _end && _left > 0) {
                        const uint64_t gap = decode_varint_bounded(_data, _pos, _end);
                        // skip zero-gap placeholders except possibly the very first emitted element
                        if (_emitted > 0 && gap == 0) { continue; }
                        _acc += static_cast<HypernodeID>(gap);
                        _cur = _acc;
                        --_left;
                        ++_emitted;
                        _has = true;
                        return;
                    }
                    _has = false;
                }

                const std::vector<uint8_t>& _data;
                size_t _end;
                size_t _pos;
                HypernodeID _acc;
                HypernodeID _cur;
                bool _has;
                size_t _left;
                size_t _emitted;
        };
        using IncidenceIterator = PinsIterator;
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
                                                                            HyperedgeID max_edge_id,
                                                                            const BitVector* enabled,
                                                                            size_t remaining)
                    : _data(data), _end(end_pos), _pos(start_pos),
                        _acc(0), _cur(0), _has(false), _max(max_edge_id), _enabled(enabled), _left(remaining), _emitted(0) {
                    advance_to_valid();
                }

                // End iterator
                IncidentNetsFilteringIterator(const std::vector<uint8_t>& data,
                                                                            size_t end_pos)
                    : _data(data), _end(end_pos), _pos(end_pos),
                        _acc(0), _cur(0), _has(false), _max(0), _enabled(nullptr), _left(0), _emitted(0) { }

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
                                                            while (_pos < _end && _left > 0) {
                        const uint64_t gap = decode_varint_bounded(_data, _pos, _end);
                                                                    // skip zero-gap placeholders except possibly the very first emitted element
                                                                    if (_emitted > 0 && gap == 0) { continue; }
                                                                    _acc += static_cast<HyperedgeID>(gap);
                                                                    // consume one logical entry regardless of enabled
                                                                    --_left; ++_emitted;
                                                                    if (_acc < _max && (_enabled == nullptr || (*_enabled)[_acc])) {
                                                                        _cur = _acc; _has = true; return;
                                                                    }
                        // else: skip invalid ID and continue
                    }
                                        // exhausted
                                        _has = false;
                }

                const std::vector<uint8_t>& _data;
                size_t _end;
                size_t _pos;
                HyperedgeID _acc;
                HyperedgeID _cur;
                bool _has;
        HyperedgeID _max;
        const BitVector* _enabled;
        size_t _left;
        size_t _emitted;
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
        _hypernode_offsets(),
        _hypernode_enabled(),
        _compressed_incident_nets(),
        _hyperedge_offsets(),
        _hyperedge_enabled(),
        _compressed_incidence_array(),
        _hypernode_weights(),
        _hyperedge_weights(),
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
        _hypernode_offsets(std::move(other._hypernode_offsets)),
        _hypernode_enabled(std::move(other._hypernode_enabled)),
        _compressed_incident_nets(std::move(other._compressed_incident_nets)),
        _hyperedge_offsets(std::move(other._hyperedge_offsets)),
        _hyperedge_enabled(std::move(other._hyperedge_enabled)),
        _compressed_incidence_array(std::move(other._compressed_incidence_array)),
        _hypernode_weights(std::move(other._hypernode_weights)),
        _hyperedge_weights(std::move(other._hyperedge_weights)),
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
            _hypernode_offsets = std::move(other._hypernode_offsets);
            _hypernode_enabled = std::move(other._hypernode_enabled);
            _compressed_incident_nets = std::move(other._compressed_incident_nets);
            _hyperedge_offsets = std::move(other._hyperedge_offsets);
            _hyperedge_enabled = std::move(other._hyperedge_enabled);
            _compressed_incidence_array = std::move(other._compressed_incidence_array);
            _hypernode_weights = std::move(other._hypernode_weights);
            _hyperedge_weights = std::move(other._hyperedge_weights);
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

    using HypernodeIterator = IDIterator<HypernodeID>;
    using HyperedgeIterator = IDIterator<HyperedgeID>;

    IteratorRange<HypernodeIterator> nodes() const {
        return IteratorRange<HypernodeIterator>(
            HypernodeIterator(this, ID(0), _num_hypernodes, /*edge*/false),
            HypernodeIterator(this, _num_hypernodes, _num_hypernodes, /*edge*/false));
    }

    IteratorRange<HyperedgeIterator> edges() const {
        return IteratorRange<HyperedgeIterator>(
            HyperedgeIterator(this, ID(0), _num_hyperedges, /*edge*/true),
            HyperedgeIterator(this, _num_hyperedges, _num_hyperedges, /*edge*/true));
    }

    IteratorRange<IncidentNetsIterator> incidentEdges(const HypernodeID u) const {
        ASSERT(nodeIsEnabled(u), "Hypernode" << u << "is disabled");
           const size_t begin = _hypernode_offsets[u];
           const size_t end = hypernode_firstInvalidEntry(u);
        size_t pos = begin;
        // read header varint (uncompressed degree)
        const size_t deg = static_cast<size_t>(decode_varint_bounded(_compressed_incident_nets, pos, end));
        return IteratorRange<IncidentNetsIterator>(
            IncidentNetsIterator(_compressed_incident_nets, pos, end, _num_hyperedges, &_hyperedge_enabled, deg),
            IncidentNetsIterator(_compressed_incident_nets, end));
    }

    IteratorRange<IncidenceIterator> pins(const HyperedgeID e) const {
        ASSERT(edgeIsEnabled(e), "Hyperedge" << e << "is disabled");
        const size_t he_begin = _hyperedge_offsets[e];
        const size_t he_end = hyperedge_firstInvalidEntry(e);
        size_t pos = he_begin;
        // read header varint (uncompressed size)
        const size_t esize = static_cast<size_t>(decode_varint_bounded(_compressed_incidence_array, pos, he_end));
        return IteratorRange<IncidenceIterator>(
            IncidenceIterator(_compressed_incidence_array, pos, he_end, esize),
            IncidenceIterator(_compressed_incidence_array, he_end));
    }

    // Expose compressed sizes (in bytes) of segments; uses trailing-zero trimming
    size_t compressedSizeOfEdge(const HyperedgeID e) const {
        const size_t begin = _hyperedge_offsets[e];
        const size_t end = hyperedge_firstInvalidEntry(e);
        return end - begin;
    }

    size_t compressedSizeOfNode(const HypernodeID u) const {
        const size_t begin = _hypernode_offsets[u];
        const size_t end = hypernode_firstInvalidEntry(u);
        return end - begin;
    }

    // Update the header varint (uncompressed size) of an edge in-place without shifting.
    // Grows only if the new varint fits into the existing header byte-length; otherwise throws.
    void setEdgeSizeHeader(const HyperedgeID e, const HypernodeID new_size) {
    ASSERT(edgeIsEnabled(e), "Hyperedge" << e << "is disabled");
    const size_t begin = _hyperedge_offsets[e];
        const size_t end = hyperedge_firstInvalidEntry(e);
        // Determine current header length
        size_t pos = begin;
        (void)decode_varint_bounded(_compressed_incidence_array, pos, end);
        const size_t cur_len = pos - begin;
        std::vector<uint8_t> header;
        encode_varint_fixed_len(static_cast<uint64_t>(new_size), cur_len, header);
        // Write header back
        std::copy(header.begin(), header.end(), _compressed_incidence_array.begin() + begin);
    }

    // Update the header varint (degree) of a node in-place without shifting.
    void setNodeDegreeHeader(const HypernodeID u, const HyperedgeID new_deg) {
    ASSERT(nodeIsEnabled(u), "Hypernode" << u << "is disabled");
    const size_t begin = _hypernode_offsets[u];
        const size_t end = hypernode_firstInvalidEntry(u);
        // Determine current header length
        size_t pos = begin;
        (void)decode_varint_bounded(_compressed_incident_nets, pos, end);
        const size_t cur_len = pos - begin;
        std::vector<uint8_t> header;
        encode_varint_fixed_len(static_cast<uint64_t>(new_deg), cur_len, header);
        // Write header back
        std::copy(header.begin(), header.end(), _compressed_incident_nets.begin() + begin);
    }

    // ####################### Hypernode Information #######################

    HypernodeWeight nodeWeight(const HypernodeID u) const {
        if (_hypernode_weights.empty()) return 1;
        return _hypernode_weights[u];
    }

    void setNodeWeight(const HypernodeID u, const HypernodeWeight weight) {
        ASSERT(nodeIsEnabled(u), "Hypernode" << u << "is disabled");
        if (_hypernode_weights.empty()) _hypernode_weights.assign(_num_hypernodes, 1);
        _hypernode_weights[u] = weight;
    }

    HyperedgeID nodeDegree(const HypernodeID u) const {
        ASSERT(nodeIsEnabled(u), "Hypernode" << u << "is disabled");
        const size_t begin = _hypernode_offsets[u];
        const size_t end = hypernode_firstInvalidEntry(u);
        size_t pos = begin;
        return static_cast<HyperedgeID>(decode_varint_bounded(_compressed_incident_nets, pos, end));
    }

    bool nodeIsEnabled(const HypernodeID u) const { return _hypernode_enabled[u]; }
    void enableHypernode(const HypernodeID u) { _hypernode_enabled[u] = true; }
    void disableHypernode(const HypernodeID u) { _hypernode_enabled[u] = false; }

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
        ASSERT(edgeIsEnabled(e), "Hyperedge" << e << "is disabled");
        if (_hyperedge_weights.empty()) return 1;
        return _hyperedge_weights[e];
    }

    void setEdgeWeight(const HyperedgeID e, const HyperedgeWeight weight) {
        ASSERT(edgeIsEnabled(e), "Hyperedge" << e << "is disabled");
        if (_hyperedge_weights.empty()) _hyperedge_weights.assign(_num_hyperedges, 1);
        _hyperedge_weights[e] = weight;
    }

    HypernodeID edgeSize(const HyperedgeID e) const {
        ASSERT(edgeIsEnabled(e), "Hyperedge" << e << "is disabled");
        const size_t begin = _hyperedge_offsets[e];
        const size_t end = hyperedge_firstInvalidEntry(e);
        size_t pos = begin;
        return static_cast<HypernodeID>(decode_varint_bounded(_compressed_incidence_array, pos, end));
    }

    HypernodeID maxEdgeSize() const {
        return _max_edge_size;
    }

    bool edgeIsEnabled(const HyperedgeID e) const { return _hyperedge_enabled[e]; }
    void enableHyperedge(const HyperedgeID e) { _hyperedge_enabled[e] = true; }
    void disableHyperedge(const HyperedgeID e) { _hyperedge_enabled[e] = false; }

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

        size_t he_pos = _hyperedge_offsets[he];
        const size_t he_end = hyperedge_firstInvalidEntry(he);
        std::vector<HypernodeID> pins_of_he;
        pins_of_he.reserve(edgeSize(he));
        {
            HypernodeID acc = 0;
            // skip header varint (edge size)
            (void)decode_varint_bounded(_compressed_incidence_array, he_pos, he_end);
            while (he_pos < he_end) {
                const uint64_t gap = decode_varint_bounded(_compressed_incidence_array, he_pos, he_end);
                acc += static_cast<HypernodeID>(gap);
                pins_of_he.push_back(acc);
            }
        }

        // For each pin, attempt in-place shrink; if any overflow, fallback to repack
        struct PendingUpdate { HypernodeID u; std::vector<uint8_t> bytes; size_t begin; size_t cap; };
        std::vector<PendingUpdate> pending; pending.reserve(pins_of_he.size());
        bool needs_repack = false;
        for (const HypernodeID u : pins_of_he) {
            if (!nodeIsEnabled(u)) continue;
            const size_t begin = _hypernode_offsets[u];
            const size_t end = hypernode_firstInvalidEntry(u);
            // Decode current incidents
            std::vector<HyperedgeID> inc; inc.reserve(nodeDegree(u));
            size_t cur = begin; (void)decode_varint_bounded(_compressed_incident_nets, cur, end); HyperedgeID acc = 0;
            while (cur < end) { const uint64_t gap = decode_varint_bounded(_compressed_incident_nets, cur, end); acc += static_cast<HyperedgeID>(gap); if (acc != he && acc < _num_hyperedges) inc.push_back(acc); }
            // Re-encode
            std::vector<uint8_t> bytes;
            std::sort(inc.begin(), inc.end()); inc.erase(std::unique(inc.begin(), inc.end()), inc.end());
            encode_varint(static_cast<uint64_t>(inc.size()), bytes); // header
            HyperedgeID prev = 0; for (HyperedgeID x : inc) { ASSERT(x >= prev); encode_varint(static_cast<uint64_t>(x - prev), bytes); prev = x; }
            // Capacity up to next enabled node
            HypernodeID v = u + 1; while (v < _num_hypernodes && !nodeIsEnabled(v)) ++v;
            size_t next_begin = (v < _num_hypernodes) ? _hypernode_offsets[v] : _compressed_incident_nets.size();
            const size_t cap = next_begin - begin;
            if (bytes.size() > cap) needs_repack = true;
            pending.push_back(PendingUpdate{u, std::move(bytes), begin, cap});
        }

        if (!needs_repack) {
            for (auto& up : pending) {
                if (!up.bytes.empty()) {
                    std::copy(up.bytes.begin(), up.bytes.end(), _compressed_incident_nets.begin() + up.begin);
                }
            }
        } else {
            // Repack only if absolutely necessary (rare due to varint edge cases)
            std::unordered_map<HypernodeID, std::vector<uint8_t>> overrides;
            overrides.reserve(pending.size());
            for (const auto& up : pending) { overrides.emplace(up.u, up.bytes); }
            // Snapshot current begins/ends
            std::vector<size_t> old_begin(_num_hypernodes), old_end(_num_hypernodes);
            for (HypernodeID u = 0; u < _num_hypernodes; ++u) { old_begin[u] = _hypernode_offsets[u]; old_end[u] = hypernode_firstInvalidEntry(u); }
            auto encode_from_snapshot = [&](HypernodeID u) {
                std::vector<HyperedgeID> inc; size_t cur = old_begin[u]; const size_t end = old_end[u];
                (void)decode_varint_bounded(_compressed_incident_nets, cur, end);
                HyperedgeID acc = 0;
                while (cur < end) { const uint64_t g = decode_varint_bounded(_compressed_incident_nets, cur, end); acc += static_cast<HyperedgeID>(g); if (acc < _num_hyperedges) inc.push_back(acc); }
                std::vector<uint8_t> out; std::sort(inc.begin(), inc.end()); inc.erase(std::unique(inc.begin(), inc.end()), inc.end());
                encode_varint(static_cast<uint64_t>(inc.size()), out);
                HyperedgeID prev = 0; for (HyperedgeID x : inc) { ASSERT(x >= prev); encode_varint(static_cast<uint64_t>(x - prev), out); prev = x; }
                return out;
            };
            CompressedIncidentNets new_incident_nets; new_incident_nets.reserve(_compressed_incident_nets.size());
            size_t write_pos = 0;
            for (HypernodeID u = 0; u < _num_hypernodes; ++u) {
                if (!nodeIsEnabled(u)) continue;
                _hypernode_offsets[u] = write_pos;
                std::vector<uint8_t> bytes;
                auto it = overrides.find(u);
                if (it != overrides.end()) { bytes = it->second; } else { bytes = encode_from_snapshot(u); }
                new_incident_nets.insert(new_incident_nets.end(), bytes.begin(), bytes.end());
                write_pos += bytes.size();
            }
            _compressed_incident_nets.swap(new_incident_nets);
        }

        ++_num_removed_hyperedges;
        disableHyperedge(he);
    }

    void removeLargeEdge(const HyperedgeID he) {
        ASSERT(edgeIsEnabled(he), "Hyperedge" << he << "is disabled");

        size_t he_pos = _hyperedge_offsets[he];
        const size_t he_end = hyperedge_firstInvalidEntry(he);
        std::vector<HypernodeID> pins_of_he;
        pins_of_he.reserve(edgeSize(he));
        {
            HypernodeID acc = 0;
            // skip header varint (edge size)
            (void)decode_varint_bounded(_compressed_incidence_array, he_pos, he_end);
            while (he_pos < he_end) {
                const uint64_t gap = decode_varint_bounded(_compressed_incidence_array, he_pos, he_end);
                acc += static_cast<HypernodeID>(gap);
                pins_of_he.push_back(acc);
            }
        }

        // Attempt in-place shrink for each affected node; fallback to repack if any overflow
        struct PendingUpdateLE { HypernodeID u; std::vector<uint8_t> bytes; size_t begin; size_t cap; };
        std::vector<PendingUpdateLE> updates; updates.reserve(pins_of_he.size());
        bool needs_repack = false;
        for (const HypernodeID u : pins_of_he) {
            if (!nodeIsEnabled(u)) continue;
            const size_t begin = _hypernode_offsets[u];
            const size_t end = hypernode_firstInvalidEntry(u);
            // Decode incidents
            std::vector<HyperedgeID> inc; inc.reserve(nodeDegree(u));
            size_t cur = begin; (void)decode_varint_bounded(_compressed_incident_nets, cur, end); HyperedgeID acc = 0;
            while (cur < end) { const uint64_t gap = decode_varint_bounded(_compressed_incident_nets, cur, end); acc += static_cast<HyperedgeID>(gap); if (acc != he && acc < _num_hyperedges) inc.push_back(acc); }
            std::vector<uint8_t> bytes;
            std::sort(inc.begin(), inc.end()); inc.erase(std::unique(inc.begin(), inc.end()), inc.end());
            encode_varint(static_cast<uint64_t>(inc.size()), bytes);
            HyperedgeID prev = 0; for (HyperedgeID x : inc) { ASSERT(x >= prev); encode_varint(static_cast<uint64_t>(x - prev), bytes); prev = x; }
            // Capacity to next enabled node
            HypernodeID v = u + 1; while (v < _num_hypernodes && !nodeIsEnabled(v)) ++v;
            const size_t next_begin = (v < _num_hypernodes) ? _hypernode_offsets[v] : _compressed_incident_nets.size();
            const size_t cap = next_begin - begin;
            if (bytes.size() > cap) needs_repack = true;
            updates.push_back(PendingUpdateLE{u, std::move(bytes), begin, cap});
        }
        if (!needs_repack) {
            for (auto& up : updates) {
                if (!up.bytes.empty()) {
                    std::copy(up.bytes.begin(), up.bytes.end(), _compressed_incident_nets.begin() + up.begin);
                }
            }
        } else {
            // Repack fallback
            std::unordered_map<HypernodeID, std::vector<uint8_t>> overrides;
            overrides.reserve(updates.size());
            for (const auto& up : updates) { overrides.emplace(up.u, up.bytes); }
            std::vector<size_t> old_begin(_num_hypernodes), old_end(_num_hypernodes);
            for (HypernodeID u = 0; u < _num_hypernodes; ++u) { old_begin[u] = _hypernode_offsets[u]; old_end[u] = hypernode_firstInvalidEntry(u); }
            auto encode_from_snapshot = [&](HypernodeID u) {
                std::vector<HyperedgeID> inc; size_t cur = old_begin[u]; const size_t end = old_end[u];
                (void)decode_varint_bounded(_compressed_incident_nets, cur, end);
                HyperedgeID acc = 0;
                while (cur < end) { const uint64_t g = decode_varint_bounded(_compressed_incident_nets, cur, end); acc += static_cast<HyperedgeID>(g); if (acc < _num_hyperedges) inc.push_back(acc); }
                std::vector<uint8_t> out; std::sort(inc.begin(), inc.end()); inc.erase(std::unique(inc.begin(), inc.end()), inc.end());
                encode_varint(static_cast<uint64_t>(inc.size()), out);
                HyperedgeID prev = 0; for (HyperedgeID x : inc) { ASSERT(x >= prev); encode_varint(static_cast<uint64_t>(x - prev), out); prev = x; }
                return out;
            };
            CompressedIncidentNets new_incident_nets; new_incident_nets.reserve(_compressed_incident_nets.size());
            size_t write_pos = 0;
            for (HypernodeID u = 0; u < _num_hypernodes; ++u) {
                if (!nodeIsEnabled(u)) continue;
                _hypernode_offsets[u] = write_pos;
                std::vector<uint8_t> bytes;
                auto it = overrides.find(u);
                if (it != overrides.end()) { bytes = it->second; } else { bytes = encode_from_snapshot(u); }
                new_incident_nets.insert(new_incident_nets.end(), bytes.begin(), bytes.end());
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
        size_t he_pos = _hyperedge_offsets[he];
        const size_t he_end = hyperedge_firstInvalidEntry(he);
        std::vector<HypernodeID> pins_of_he;
        pins_of_he.reserve(edgeSize(he));
        {
            HypernodeID acc = 0;
            // skip header varint (edge size)
            (void)decode_varint_bounded(_compressed_incidence_array, he_pos, he_end);
            while (he_pos < he_end) {
                const uint64_t gap = decode_varint_bounded(_compressed_incidence_array, he_pos, he_end);
                acc += static_cast<HypernodeID>(gap);
                pins_of_he.push_back(acc);
            }
        }

        // First pass: compute new bytes per affected node and check if in-place growth fits
    struct PendingUpdate { HypernodeID u; std::vector<uint8_t> bytes; size_t begin; size_t cap; };
        std::vector<PendingUpdate> pending; pending.reserve(pins_of_he.size());
        bool needs_repack = false;
        for (const HypernodeID u : pins_of_he) {
            if (!nodeIsEnabled(u)) continue;
            const size_t begin = _hypernode_offsets[u];
            const size_t end = hypernode_firstInvalidEntry(u);
            // Decode existing incidents
            std::vector<HyperedgeID> inc; inc.reserve(nodeDegree(u) + 1);
            size_t cur = begin; (void)decode_varint_bounded(_compressed_incident_nets, cur, end); HyperedgeID acc = 0;
            while (cur < end) { const uint64_t g = decode_varint_bounded(_compressed_incident_nets, cur, end); acc += static_cast<HyperedgeID>(g); if (acc < _num_hyperedges) inc.push_back(acc); }
            // Insert he and encode
            inc.push_back(he);
            std::sort(inc.begin(), inc.end()); inc.erase(std::unique(inc.begin(), inc.end()), inc.end());
            std::vector<uint8_t> bytes;
            encode_varint(static_cast<uint64_t>(inc.size()), bytes);
            HyperedgeID prev = 0; for (HyperedgeID x : inc) { ASSERT(x >= prev); encode_varint(static_cast<uint64_t>(x - prev), bytes); prev = x; }
            // Compute available capacity up to next enabled node begin (or vector end)
            HypernodeID v = u + 1; while (v < _num_hypernodes && !nodeIsEnabled(v)) ++v;
            size_t next_begin = (v < _num_hypernodes) ? _hypernode_offsets[v] : _compressed_incident_nets.size();
            size_t cap = next_begin - begin; // bytes we can occupy without moving neighbors
            if (bytes.size() > cap) needs_repack = true;
            pending.push_back(PendingUpdate{u, std::move(bytes), begin, cap});
        }

        if (!needs_repack) {
            // In-place apply all updates (may extend vector for last node)
            for (auto& up : pending) {
                size_t needed_end = up.begin + up.bytes.size();
                if (needed_end > _compressed_incident_nets.size()) {
                    _compressed_incident_nets.resize(needed_end);
                }
                if (!up.bytes.empty()) {
                    std::copy(up.bytes.begin(), up.bytes.end(), _compressed_incident_nets.begin() + up.begin);
                }
            }
        } else {
            // Fallback: single repack using prepared bytes for affected nodes
            std::unordered_map<HypernodeID, std::vector<uint8_t>> overrides;
            overrides.reserve(pending.size());
            for (const auto& up : pending) { overrides.emplace(up.u, up.bytes); }
            // Snapshot current begins/ends
            std::vector<size_t> old_begin(_num_hypernodes), old_end(_num_hypernodes);
            for (HypernodeID u = 0; u < _num_hypernodes; ++u) { old_begin[u] = _hypernode_offsets[u]; old_end[u] = hypernode_firstInvalidEntry(u); }
            // Helper to decode an untouched node from snapshot and encode again
            auto encode_from_snapshot = [&](HypernodeID u) {
                std::vector<HyperedgeID> inc; size_t cur = old_begin[u]; const size_t end = old_end[u];
                (void)decode_varint_bounded(_compressed_incident_nets, cur, end);
                HyperedgeID acc = 0;
                while (cur < end) { const uint64_t g = decode_varint_bounded(_compressed_incident_nets, cur, end); acc += static_cast<HyperedgeID>(g); if (acc < _num_hyperedges) inc.push_back(acc); }
                std::vector<uint8_t> out; if (!inc.empty()) { std::sort(inc.begin(), inc.end()); inc.erase(std::unique(inc.begin(), inc.end()), inc.end()); encode_varint(static_cast<uint64_t>(inc.size()), out); HyperedgeID prev = 0; for (HyperedgeID x : inc) { ASSERT(x >= prev); encode_varint(static_cast<uint64_t>(x - prev), out); prev = x; } }
                return out;
            };
            CompressedIncidentNets new_incident_nets; new_incident_nets.reserve(_compressed_incident_nets.size() + 16);
            size_t write_pos = 0;
            for (HypernodeID u = 0; u < _num_hypernodes; ++u) {
                if (!nodeIsEnabled(u)) continue;
                _hypernode_offsets[u] = write_pos;
                std::vector<uint8_t> bytes;
                auto it = overrides.find(u);
                if (it != overrides.end()) {
                    bytes = it->second;
                } else {
                    bytes = encode_from_snapshot(u);
                }
                new_incident_nets.insert(new_incident_nets.end(), bytes.begin(), bytes.end());
                write_pos += bytes.size();
            }
            _compressed_incident_nets.swap(new_incident_nets);
            // No explicit size metadata to update
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
    _hypernode_offsets.clear();
    _hypernode_offsets.shrink_to_fit();
    _hypernode_enabled.clear();
    _hypernode_enabled.shrink_to_fit();
        _compressed_incident_nets.clear();
        _compressed_incident_nets.shrink_to_fit();
    _hyperedge_offsets.clear();
    _hyperedge_offsets.shrink_to_fit();
    _hyperedge_enabled.clear();
    _hyperedge_enabled.shrink_to_fit();
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
        total += sizeof(size_t) * _hypernode_offsets.capacity();
    // BitVector packs bits; approximate bytes as ceil(capacity_in_bits/8)
        total += ((_hypernode_enabled.capacity() + 7) / 8);
        total += sizeof(uint8_t) * _compressed_incident_nets.size();
        total += sizeof(size_t) * _hyperedge_offsets.capacity();
    total += ((_hyperedge_enabled.capacity() + 7) / 8);
        total += sizeof(uint8_t) * _compressed_incidence_array.size();
        total += sizeof(HypernodeWeight) * _hypernode_weights.capacity();
        total += sizeof(HyperedgeWeight) * _hyperedge_weights.capacity();
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

    MT_KAHYPAR_ATTRIBUTE_ALWAYS_INLINE Hypernode hypernode(const HypernodeID u) const {
        ASSERT(u < _num_hypernodes, "Hypernode" << u << " does not exist");
        return Hypernode(this, u);
    }

    MT_KAHYPAR_ATTRIBUTE_ALWAYS_INLINE Hypernode hypernode(const HypernodeID u) {
        return Hypernode(this, u);
    }

   MT_KAHYPAR_ATTRIBUTE_ALWAYS_INLINE IteratorRange<IncidentNetsIterator>
   incident_nets_of(const HypernodeID u, const size_t pos = 0) const {
       ASSERT(!hypernode(u).isDisabled(), "Hypernode" << u << "is disabled");
       if (pos != 0) {
           throw UnsupportedOperationException(
               "incident_nets_of(u, pos) with pos > 0 is not supported in compressed hypergraph");
        }
       const size_t begin = hypernode(u).firstEntry();
       const size_t end = hypernode_firstInvalidEntry(u);
       size_t p = begin; const size_t deg = static_cast<size_t>(decode_varint_bounded(_compressed_incident_nets, p, end));
       return IteratorRange<IncidentNetsIterator>(
           IncidentNetsIterator(_compressed_incident_nets, p, end, _num_hyperedges, &_hyperedge_enabled, deg),
           IncidentNetsIterator(_compressed_incident_nets, end));
   }

    // ####################### Hyperedge Information #######################

    MT_KAHYPAR_ATTRIBUTE_ALWAYS_INLINE Hyperedge hyperedge(const HyperedgeID e) const {
        ASSERT(e < _num_hyperedges, "Hyperedge " << e << " does not exist");
        return Hyperedge(this, e);
    }

    MT_KAHYPAR_ATTRIBUTE_ALWAYS_INLINE Hyperedge hyperedge(const HyperedgeID e) {
        return Hyperedge(this, e);
    }

    MT_KAHYPAR_ATTRIBUTE_ALWAYS_INLINE HypernodeID pinAt(const HyperedgeID e, const size_t local_pos) const {
        ASSERT(edgeIsEnabled(e), "Hyperedge" << e << "is disabled");
        ASSERT(local_pos < edgeSize(e));
        // Decode varints up to local_pos, skipping zero-gap placeholders
        size_t cur = _hyperedge_offsets[e];
        const size_t end = hyperedge_firstInvalidEntry(e);
        [[maybe_unused]] const size_t esize = static_cast<size_t>(decode_varint_bounded(_compressed_incidence_array, cur, end));
        ASSERT(local_pos < esize);
        HypernodeID acc = 0;
        size_t emitted = 0;
        while (cur < end && emitted <= local_pos) {
            const uint64_t gap = decode_varint_bounded(_compressed_incidence_array, cur, end);
            if (emitted > 0 && gap == 0) { continue; }
            acc += static_cast<HypernodeID>(gap);
            if (emitted == local_pos) return acc;
            ++emitted;
        }
        // Should not reach here due to assertion
        return acc;
    }

    // Computes end offset for hyperedge e using neighbor begin or array end; then trims trailing zeros.
    MT_KAHYPAR_ATTRIBUTE_ALWAYS_INLINE size_t hyperedge_firstInvalidEntry(const HyperedgeID e) const {
        ASSERT(e < _num_hyperedges);
        const size_t end = (e + 1 < _num_hyperedges) ? _hyperedge_offsets[e + 1] : _compressed_incidence_array.size();
        return end;
    }

    // Computes end offset for hypernode u using neighbor begin or array end; then trims trailing zeros.
    MT_KAHYPAR_ATTRIBUTE_ALWAYS_INLINE size_t hypernode_firstInvalidEntry(const HypernodeID u) const {
        ASSERT(u < _num_hypernodes);
        const size_t end = (u + 1 < _num_hypernodes) ? _hypernode_offsets[u + 1] : _compressed_incident_nets.size();
        return end;
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
    std::vector<size_t> _hypernode_offsets;   // CSR offsets into _compressed_incident_nets
    BitVector _hypernode_enabled;             // true = enabled, false = disabled
    CompressedIncidentNets _compressed_incident_nets;    // Compressed incident nets storage
    std::vector<size_t> _hyperedge_offsets;   // CSR offsets into _compressed_incidence_array
    BitVector _hyperedge_enabled;             // true = enabled, false = disabled
    CompressedIncidenceArray _compressed_incidence_array;  // Compressed pins storage
    // Lazy weight storage (default 1 when empty)
    std::vector<HypernodeWeight> _hypernode_weights;
    std::vector<HyperedgeWeight> _hyperedge_weights;

    // Communities and fixed vertices (uncompressed as requested)
    ds::Clustering _community_ids;
    FixedVertexSupport<CompressedHypergraph> _fixed_vertices;

    // Data reused throughout multilevel hierarchy
    TmpContractionBuffer* _tmp_contraction_buffer = nullptr;
};

} // namespace ds
} // namespace mt_kahypar
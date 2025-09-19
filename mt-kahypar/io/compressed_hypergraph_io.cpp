#include "compressed_hypergraph_io.h"
#include "mt-kahypar/datastructures/compressed_hypergraph.h"

namespace mt_kahypar {
namespace io {

mt_kahypar_hypergraph_t streamAndCompressHypergraphFile(
  const std::string& filename,
  const bool remove_single_pin_hes,
  bool /*stable*/) {

  using CHG = ds::CompressedHypergraph;
  CHG* hg = new CHG(filename, remove_single_pin_hes);
  return { reinterpret_cast<mt_kahypar_hypergraph_s*>(hg), CHG::TYPE };
}

mt_kahypar_hypergraph_t streamAndCompressGraphFile(
  const std::string& filename,
  bool stable) {
  return streamAndCompressHypergraphFile(filename, /*remove_single_pin_hes=*/false, stable);
}

} // namespace io
} // namespace mt_kahypar

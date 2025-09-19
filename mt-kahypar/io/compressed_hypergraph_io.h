#include "mt-kahypar/datastructures/compressed_hypergraph.h"

namespace mt_kahypar {
namespace io {

mt_kahypar_hypergraph_t streamAndCompressHypergraphFile(
  const std::string& filename,
  const bool remove_single_pin_hes,
  bool /*stable*/);

mt_kahypar_hypergraph_t streamAndCompressGraphFile(
  const std::string& filename,
  bool stable);
} // namespace io
} // namespace mt_kahypar

/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

  This file is part of PIPS.

  PIPS is free software: you can redistribute it and/or modify it
  under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with PIPS.  If not, see <http://www.gnu.org/licenses/>.

*/

/*
 * data structures that describe FREIA Terapix target.
 */

#ifndef HWAC_FREIA_TERAPIX_H_
#define HWAC_FREIA_TERAPIX_H_

// terapix specific property names
#define trpx_mem_prop     "HWAC_TERAPIX_RAMPE"
#define trpx_npe_prop     "HWAC_TERAPIX_NPE"
#define trpx_dmabw_prop   "HWAC_TERAPIX_DMABW"
#define trpx_gram_width   "HWAC_TERAPIX_GRAM_WIDTH"
#define trpx_gram_height  "HWAC_TERAPIX_GRAM_HEIGHT"
#define trpx_dag_cut      "HWAC_TERAPIX_DAG_CUT"
#define trpx_overlap_io   "HWAC_TERAPIX_OVERLAP_IO"
#define trpx_max_size     "HWAC_TERAPIX_IMAGELET_MAX_SIZE"

// various dag cutting strategies for terapix
#define trpx_dag_cut_none_p(s) same_string_p(s, "none")
#define trpx_dag_cut_compute_p(s) same_string_p(s, "compute")
#define trpx_dag_cut_enumerate_p(s) same_string_p(s, "enumerate")
#define trpx_dag_cut_is_valid(s) \
  trpx_dag_cut_none_p(s) ||      \
  trpx_dag_cut_compute_p(s) ||   \
  trpx_dag_cut_enumerate_p(s)

#define trpx_overlap_io_p() get_bool_property(trpx_overlap_io)

// includes for generated terapix helper
#define FREIA_TRPX_INCLUDES           \
  "// freia terapix includes\n"       \
  "#include <freiaCommon.h>\n"        \
  "#include <freiaMediumGrain.h>\n"		\
  "#include <freiaCoarseGrain.h>\n"		\
  "#include <terapix.h>\n"            \
  "#include <terapix_ucode.h>\n"

// terapix freia function description
typedef struct {
  // image erosion in each direction
  int north, south, east, west;
  // internal memory needed, in pixels (?)
  int memory;
  // approximate cost per pixel per row
  // this information is available in the terapix simulator
  int cost;
  // whether it can be done in place
  bool inplace;
  // whether the input images order must be reversed when calling the hardware
  bool reverse;
  // microcode segment name
  string ucode;
} terapix_hw_t;

#endif /* !HWAC_FREIA_TERAPIX_H_ */

/*

  $Id$

  Copyright 1989-2010 MINES ParisTech

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

#define trpx_mem_prop "HWAC_TERAPIX_RAMPE"
#define trpx_npe_prop "HWAC_TERAPIX_NPE"
#define trpx_dmabw_prop "HWAC_TERAPIX_DMABW"
#define trpx_gram_width "HWAC_TERAPIX_GRAM_WIDTH"
#define trpx_gram_height "HWAC_TERAPIX_GRAM_HEIGHT"

#define FREIA_TRPX_INCLUDES			\
  "#include <freiaCommon.h>\n"			\
  "#include <freiaMediumGrain.h>\n"		\
  "#include <freiaCoarseGrain.h>\n"		\
  "#include <terapix.h>\n"			\
  "#include <terapix_ucode.h>\n"

typedef struct {
  // image erosion in each direction
  int north, south, east, west;
  // internal memory needed, in pixels (?)
  int memory;
  // approximate cost per pixel per row
  int cost;
  // whether it can be done in place
  boolean inplace;
  // code segment name
  string ucode;
} terapix_hw_t;

#endif /* !HWAC_FREIA_TERAPIX_H_ */

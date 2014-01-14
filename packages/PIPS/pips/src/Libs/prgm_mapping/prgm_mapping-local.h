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

#define MAPPING_MODULE_NAME "MAPPING"

#define CONST_COEFF "CC"
#define INDEX_COEFF "XC"
#define PARAM_COEFF "PC"
#define AUXIL_COEFF "AC"
#define LAMBD_COEFF "LC"
#define MU_COEFF    "ZM"
#define INDEX_VARIA "XV"
#define DIFFU_COEFF "DC"


#define VERTEX_DOMAIN(v) \
	dfg_vertex_label_exec_domain((dfg_vertex_label) vertex_vertex_label(v))

#define SUCC_DATAFLOWS(s) \
	dfg_arc_label_dataflows((dfg_arc_label) successor_arc_label(s))


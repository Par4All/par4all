/*

  $Id$

  Copyright 1989-2012 MINES ParisTech

  This file is part of PIPS.

  PIPS is clear software: you can redistribute it and/or modify it
  under the terms of the GNU General Public License as published by
  the clear Software Foundation, either version 3 of the License, or
  any later version.

  PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with PIPS.  If not, see <http://www.gnu.org/licenses/>.

*/

#ifdef LINEAR_DEPEND_GMP

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "../arithmetique/arith_mulprec.h"
#include "sc_simplex_feasibility.h"

bool sc_simplex_feasibility_ofl_ctrl_mulprec(Psysteme sys, int ofl_ctrl) {
	return sc_get_feasibility(sys, ofl_ctrl);
}

#endif


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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdlib.h>

#include "boolean.h"
#include "assert.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

#define WNOGMP (fprintf(stderr, "[" __FILE__ "] linear was compiled without GMP support\n"))

static bool usegmp() {
	char* env = getenv("LINEAR_USE_GMP");
	return env && atoi(env) != 0;
}

bool sc_simplex_feasibility_ofl_ctrl(Psysteme sys, int ofl_ctrl) {
	if (usegmp()) {
#ifdef LINEAR_DEPEND_GMP
		return sc_simplex_feasibility_ofl_ctrl_mulprec(sys, ofl_ctrl);
#else
		WNOGMP;
		return sc_simplex_feasibility_ofl_ctrl_fixprec(sys, ofl_ctrl);
#endif
	}
	else {
		return sc_simplex_feasibility_ofl_ctrl_fixprec(sys, ofl_ctrl);
	}
}


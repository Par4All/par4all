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

/**
 * @file
 * This file provides a function to test whether a constraint system is
 * feasible, using simplex method, in multiple precision.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifdef HAVE_GMP_H

#include "../arithmetique/arith_mulprec.h"
#include "sc_simplex_feasibility.h"

/**
 * Main Function
 */
/**@{*/

/**
 * Determine whether a system @a sys of equations and inequations is feasible.
 * Parameter @a ofl_ctrl indicates whether an overflow control is performed
 * (possible values: @c NO_OFL_CTRL, @c FWD_OFL_CTRL).
 * Since we are in multiple precision, this parameter is ignored.
 */
bool sc_simplex_feasibility_ofl_ctrl_mulprec(Psysteme sys, int ofl_ctrl)
{
	return sc_get_feasibility(sys, ofl_ctrl);
}

/**@}*/

#endif


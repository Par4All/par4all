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
#include "reductions_private.h"
/* shorthands for REDUCTION:
 */
#define reduction_variable(r) reference_variable(reduction_reference(r))
#define reduction_none_p(r) reduction_operator_none_p(reduction_op(r))
#define reduction_tag(r) reduction_operator_tag(reduction_op(r))
#define make_none_reduction(var) \
    make_reduction(make_reference(var, NIL), \
                   make_reduction_operator(is_reduction_operator_none, UU),\
                   NIL, NIL)

/* quick debug macros
 */
#define DEBUG_REDUCTION(level, msg, red) \
  ifdebug(level){pips_debug(level, msg); print_reduction(red);}
#define DEBUG_REDUCTIONS(level, msg, reds) \
  ifdebug(level){pips_debug(level, msg); \
                 gen_map(print_reduction, reductions_list(reds));}

/* end of it
 */




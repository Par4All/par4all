/*

  $Id: ri-util-local.h 17253 2010-05-31 08:43:19Z creusillet $

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

#include "linear.h"
#include "newgen.h"
#include "ri.h"

#define effect_system(e) \
        (descriptor_convex_p(effect_descriptor(e))? \
         descriptor_convex(effect_descriptor(e)) : SC_UNDEFINED)

/* FI: it would be useful to assert cell_preference_p(effect_cell(e)),
   but I do not know how to do it in such a way that it works both for
   left hand sides and right hand sides using commas.
   I definitely remove this one : it is too dangerous.
*/
/* #define effect_reference(e)					\
   preference_reference(cell_preference(effect_cell(e))) */
#define effect_reference(e) \
  pips_internal_error("effect_reference not defined anymore \n")

/* FI: cannot be used as a left hand side */
#define effect_any_reference(e) \
         (cell_preference_p(effect_cell(e))? preference_reference(cell_preference(effect_cell(e))) : cell_reference(effect_cell(e)))
#define make_preference_simple_effect(reference,action,approximation)\
    make_effect(make_cell(is_cell_preference, make_preference(reference)),\
		(action), (approximation),	\
		make_descriptor(is_descriptor_none,UU))

#define make_reference_simple_effect(reference,action,approximation)\
  make_effect(make_cell(is_cell_reference, (reference)),	    \
		(action), (approximation),	\
		make_descriptor(is_descriptor_none,UU))

#define make_simple_effect(reference,action,approximation)\
    make_effect(make_cell(is_cell_preference, make_preference(reference)),\
		(action), (approximation),	\
		make_descriptor(is_descriptor_none,UU))

#define make_convex_effect(reference,action,approximation,system)\
  make_effect(make_cell(is_reference, (reference)),			\
		(action), (approximation),				\
		make_descriptor(is_descriptor_convex,system))


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

/* actions interpretation for simple effects prettyprinting */
typedef enum {act_rw, act_inout, act_live_in, act_live_out}
  simple_effects_actions_interpretations;

/* copies an effect with no subcript expression */
#define make_sdfi_effect(e) \
 (reference_indices(effect_any_reference(e)) == NIL) ? \
  make_simple_effect(make_reference(reference_variable(effect_any_reference(e)), NIL),\
     copy_action(effect_action(e)), \
     make_approximation(approximation_tag(effect_approximation(e)), UU)) : \
  make_simple_effect(make_reference(reference_variable(effect_any_reference(e)), NIL), \
	      copy_action(effect_action(e)), \
	      make_approximation(is_approximation_may, UU))

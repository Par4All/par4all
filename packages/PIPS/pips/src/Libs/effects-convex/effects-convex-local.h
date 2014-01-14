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

/* simulation of the type region */

#define region effect
#define REGION EFFECT  /* for casts */
#define region_CAST REGION_CAST
#define REGION_CAST(x) REGION(x)
#define REGION_ EFFECT_  /* for some left casts */
#define REGION_TYPE EFFECT_TYPE
#define REGION_NEWGEN_DOMAIN EFFECT_NEWGEN_DOMAIN
#define gen_REGION_cons gen_effect_cons
#define gen_region_cons gen_effect_cons

#define region_undefined effect_undefined
#define region_undefined_p(reg)  effect_undefined_p((reg))
#define make_region(reference,action,approximation,system) \
  make_effect(make_cell(is_cell_reference, (reference)),   \
		(action), (approximation),	\
		make_descriptor(is_descriptor_convex,system))
/* To be avoided. Use region_any_reference() instead.
 I definitely remove this one : it is too dangerous. BC.
*/
/* #define region_reference(reg)					\
    preference_reference(cell_preference(effect_cell(reg)))
*/
#define region_reference(reg) \
  pips_error("region_reference is not defined anymore \n")
#define region_any_reference(reg) \
  (cell_preference_p(effect_cell(reg)) ? preference_reference(cell_preference(effect_cell(reg))) : cell_reference(effect_cell(reg)))
#define region_action(reg) effect_action(reg)
#define region_approximation(reg) effect_approximation(reg)
#define region_context(reg) effect_context(reg)
#define region_cell(reg) effect_cell(reg)

#define region_entity(reg) reference_variable(region_any_reference(reg))
#define region_action_tag(reg) action_tag(effect_action(reg))
#define region_approximation_tag(reg) \
    approximation_tag(effect_approximation(reg))
#define region_system_(reg) \
    descriptor_convex_(effect_descriptor(reg))
#define region_system(reg) \
    descriptor_convex(effect_descriptor(reg))

/* useful region macros */

#define region_read_p(reg) (action_tag(effect_action(reg))==is_action_read)
#define region_write_p(reg) (action_tag(effect_action(reg))==is_action_write)
#define region_may_p(reg) \
        (approximation_tag(effect_approximation(reg)) == is_approximation_may)
#define region_must_p(reg) \
        (approximation_tag(effect_approximation(reg)) == is_approximation_must)
#define region_exact_p(reg) \
        (approximation_tag(effect_approximation(reg)) == is_approximation_exact)
#define region_empty_p(reg) sc_empty_p(region_system(reg))
#define region_rn_p(reg) sc_rn_p(region_system(reg))
#define region_scalar_p(reg) (anywhere_effect_p(reg) || entity_scalar_p(region_entity(reg)))
#define effect_region_p(e) (descriptor_convex_p(effect_descriptor(e)))

/* consistency checking */
#define debug_regions_consistency(l_reg)\
ifdebug(1){regions_consistent_p(l_reg);}
#define debug_region_consistency(reg)\
ifdebug(1){region_consistent_p(reg);}

/* convenient debug messages */
#define debug_print_region(level, message, region) \
  ifdebug(level) { pips_debug(level, "%s\n", message); print_region(region);}

/* other things */



#define R_RW 0
#define R_IN 1
#define R_OUT 2

#define SEQUENTIAL_REGION_SUFFIX ".reg"
#define SEQUENTIAL_PROPER_REGION_SUFFIX ".preg"
#define USER_REGION_SUFFIX ".ureg"
#define SEQUENTIAL_IN_REGION_SUFFIX ".inreg"
#define USER_IN_REGION_SUFFIX ".uinreg"
#define SEQUENTIAL_OUT_REGION_SUFFIX ".outreg"
#define USER_OUT_REGION_SUFFIX ".uoutreg"

#define NB_MAX_ARRAY_DIM 12

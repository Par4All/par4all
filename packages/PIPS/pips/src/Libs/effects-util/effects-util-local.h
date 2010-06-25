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
#include "effects.h"
#include "points_to_private.h"

/* The definition of *ANYWHERE*:*ANYWHERE* should be removed from
   ri-util-local.h */
#define ANY_MODULE_NAME "*ANY_MODULE*"
#define ANYWHERE_LOCATION "*ANYWHERE*"
#define NOWHERE_LOCATION "*NOWHERE*"
#define NULL_POINTER_NAME "*NULL_POINTER*"

#define PHI_PREFIX "PHI"
#define PSI_PREFIX "PSI"
#define RHO_PREFIX "RHO"
#define BETA_PREFIX "BETA"
#define PROPER TRUE
#define SUMMARY FALSE

/* some useful SHORTHANDS for EFFECT:
 */
/* FI: Let's hope this one is not used as lhs! */
#define effect_any_entity(e) reference_variable(effect_any_reference(e))
#define effect_action_tag(eff) action_tag(effect_action(eff))
#define effect_approximation_tag(eff) \
	approximation_tag(effect_approximation(eff))

/* #define effect_scalar_p(eff) entity_scalar_p(effect_entity(eff))
 *
 * The semantics of effects_scalar_p() must be refined. If all the
 * indices are constant expressions, we still have a scalar effect,
 * unless they are later replaced by "*", as is the case currently for
 * summary effects.
 *
 * Potential bug: eff is evaluated twice. Should be copied in a local
 * variable and braces be used.
 */
#define effect_scalar_p(eff) ((type_depth(entity_type(effect_entity(eff)))==0) \
			      && ENDP(reference_indices(effect_any_reference(eff))))
#define effect_read_p(eff) (action_tag(effect_action(eff))==is_action_read)
#define effect_write_p(eff) (action_tag(effect_action(eff))==is_action_write)
#define effect_may_p(eff) \
        (approximation_tag(effect_approximation(eff)) == is_approximation_may)
#define effect_must_p(eff) \
        (approximation_tag(effect_approximation(eff)) == is_approximation_must)
#define effect_exact_p(eff) \
        (approximation_tag(effect_approximation(eff)) ==is_approximation_exact)




/* For COMPATIBILITY purpose only - DO NOT USE anymore
 */
#define effect_variable(e) reference_variable(effect_any_reference(e))



/* TRUE if e is a phi variable
 * PHI entities have a name like: REGIONS:PHI#, where # is a number.
 * takes care if TCST and undefined entities, just in case.
 * FC, 09/12/94
 */
#define variable_phi_p(e) \
  ((e)!=(entity)NULL && (e)!=entity_undefined && \
    strncmp(entity_name(e), REGIONS_MODULE_NAME, 10)==0 && \
    strstr(entity_name(e), PHI_PREFIX) != NULL)

#define variable_psi_p(e) \
  ((e)!=(entity)NULL && (e)!=entity_undefined && \
    strncmp(entity_name(e), REGIONS_MODULE_NAME, 10)==0 && \
    strstr(entity_name(e), PSI_PREFIX) != NULL)

#define variable_rho_p(e) \
  ((e)!=(entity)NULL && (e)!=entity_undefined && \
    strncmp(entity_name(e), REGIONS_MODULE_NAME, 10)==0 && \
    strstr(entity_name(e), RHO_PREFIX) != NULL)

#define variable_beta_p(e)\
  ((e)!=(entity)NULL && (e)!=entity_undefined && \
    strncmp(entity_name(e), REGIONS_MODULE_NAME, 10)==0 && \
    strstr(entity_name(e), BETA_PREFIX) != NULL)

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


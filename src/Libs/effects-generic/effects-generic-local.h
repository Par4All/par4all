/*

  $Id$

  Copyright 1989-2009 MINES ParisTech

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

/* some useful SHORTHANDS for EFFECT:
 */
/* FI: Let's hope this one is not used as lhs! */
#define effect_entity(e) reference_variable(effect_any_reference(e))
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


/* some string constants for prettyprints...
 */
#define ACTION_UNDEFINED 	string_undefined
#define ACTION_READ 		"R"
#define ACTION_WRITE 		"W"
#define ACTION_IN    		"IN"
#define ACTION_OUT		"OUT"
#define ACTION_COPYIN		"COPYIN"
#define ACTION_COPYOUT		"COPYOUT"
#define ACTION_PRIVATE		"PRIVATE"


/* prettyprint function types:
 */
#include "text.h" /* hum... */
typedef text (*generic_text_function)(list /* of effect */);
typedef void (*generic_prettyprint_function)(list /* of effect */);
typedef void (*generic_attachment_function)(text);

 
/* for db_* functions 
 */
#define DB_GET_SE(name, NAME)				\
static statement_effects db_get_##name(char * modname)	\
{ return (statement_effects)				\
  db_get_memory_resource(DBR_##NAME, modname, TRUE);}

#define DB_GET_LS(name, NAME)				\
static list db_get_##name(char * modname)		\
{ return effects_to_list((effects)			\
  db_get_memory_resource(DBR_##NAME, modname, TRUE));}

#define DB_PUT_SE(name, NAME)						\
static void db_put_##name(char * modname, statement_effects se)		\
{ DB_PUT_MEMORY_RESOURCE(DBR_##NAME, modname, (char*) se);}

#define DB_PUT_LS(name, NAME)				\
static void db_put_##name(char * modname, list l)	\
{DB_PUT_MEMORY_RESOURCE(DBR_##NAME,modname,(char*)list_to_effects(l));}

#define DB_NOPUT_SE(name)\
static void db_put_##name(char *m, statement_effects se) \
{ free_statement_effects(se); return; }

#define DB_NOPUT_LS(name)\
static void db_put_##name(char *m, list l) \
{ gen_full_free_list(l); return;}

#define DB_GETPUT_SE(name, NAME) DB_GET_SE(name, NAME) DB_PUT_SE(name, NAME)
#define DB_GETNOPUT_SE(name, NAME) DB_GET_SE(name, NAME) DB_NOPUT_SE(name)
#define DB_GETPUT_LS(name, NAME) DB_GET_LS(name, NAME) DB_PUT_LS(name, NAME)
#define DB_GETNOPUT_LS(name, NAME) DB_GET_LS(name, NAME)DB_NOPUT_LS(name)


/* for debug 
*/
#define pips_debug_effect(level, message, eff) \
  ifdebug(level) { pips_debug(level, "%s\n", message); \
  (*effect_consistent_p_func)(eff); \
  (*effect_prettyprint_func)(eff);}

#define pips_debug_effects(level, message, l_eff) \
  ifdebug(level) { pips_debug(level, "%s\n", message); \
  generic_print_effects(l_eff);}




/* For COMPATIBILITY purpose only - DO NOT USE anymore
 */
#define effect_variable(e) reference_variable(effect_any_reference(e))

/* end of effects-generic-local.h
 */

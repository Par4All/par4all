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
#include "effects.h"



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
#define ACTION_LIVE_IN		"ALIVE (IN)"
#define ACTION_LIVE_OUT		"ALIVE (OUT)"

/* for debug 
*/
#define pips_debug_effect(level, message, eff) \
  ifdebug(level) { pips_debug(level, "%s\n", message); \
  (*effect_consistent_p_func)(eff); \
  (*effect_prettyprint_func)(eff);}

#define pips_debug_effects(level, message, l_eff) \
  ifdebug(level) { pips_debug(level, "%s\n", message); \
  generic_print_effects(l_eff);}

/* prettyprint function types:
 */
#include "text.h" /* hum... */
typedef text (*generic_text_function)(list /* of effect */);
typedef void (*generic_prettyprint_function)(list /* of effect */);
typedef void (*generic_attachment_function)(text);

 
/* for db_* functions 
 */
#define DB_GET_SE(name, NAME)				\
static statement_effects db_get_##name(const char *module_name)	\
{ return (statement_effects)				\
  db_get_memory_resource(DBR_##NAME, module_name, true);}

#define DB_GET_LS(name, NAME)				\
static list db_get_##name(const char *module_name)		\
{ return effects_to_list((effects)			\
  db_get_memory_resource(DBR_##NAME, module_name, true));}

#define DB_PUT_SE(name, NAME)						\
static void db_put_##name(const char *module_name, statement_effects se)		\
{ DB_PUT_MEMORY_RESOURCE(DBR_##NAME, module_name, (char*) se);}

#define DB_PUT_LS(name, NAME)				\
static void db_put_##name(const char *module_name, list l)	\
{DB_PUT_MEMORY_RESOURCE(DBR_##NAME,module_name,(char*)list_to_effects(l));}

#define DB_NOPUT_SE(name)\
static void db_put_##name(const char *m, statement_effects se) \
{ free_statement_effects(se); return; }

#define DB_NOPUT_LS(name)\
static void db_put_##name(const char *m, list l) \
{ gen_full_free_list(l); return;}

#define DB_GETPUT_SE(name, NAME) DB_GET_SE(name, NAME) DB_PUT_SE(name, NAME)
#define DB_GETNOPUT_SE(name, NAME) DB_GET_SE(name, NAME) DB_NOPUT_SE(name)
#define DB_GETPUT_LS(name, NAME) DB_GET_LS(name, NAME) DB_PUT_LS(name, NAME)
#define DB_GETNOPUT_LS(name, NAME) DB_GET_LS(name, NAME)DB_NOPUT_LS(name)

typedef enum {with_no_pointer_info, with_points_to, with_pointer_values} pointer_info_val;

typedef enum {simple, convex} effects_representation_val;

/* end of effects-generic-local.h
 */

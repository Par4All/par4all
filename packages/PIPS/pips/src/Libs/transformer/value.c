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
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
 /* VARIABLE VALUE MANAGEMENT PACKAGE FOR TRANSFORMERS
  *
  * Values of variables could be a new data structure or a special kind of
  * entities.  To avoid an increase in data structure declarations and an
  * increased size for the implicit array of entities, values are often
  * entities with special names. Their meaning is defined with respect to
  * an environment defined at the module level and so, constant during the
  * analysis of a module.
  *
  * Constant values may be used as values. They are 0-ary functions. They
  * are used to encode character strings. They are not dealt with by this
  * package. It is possible to handle scalar floating point
  * constants in the same way.
  *
  * Temporary values may be used to denote expression values. These
  * temporary values have no names. They are read only. They must be
  * eliminated from long lasting data structures as their semantics is
  * local to a single expression, usually a right hand side. Only the most
  * basic entity printing routine can be used for such values.
  *
  * Only scalar variables are analyzed.
  *
  * Properties are used to select the variables types that are to be
  * analyzed among integer, boolean, string, float and/or complex. Any
  * combination of types is legal. Scalar variables of an analyzed type
  * are said to be analyzable but they may be not analyzed because of some
  * static aliasing (EQUIVALENCE).
  *
  * Three different kind of variable values are used: new, old and
  * intermediate.  New values are used in post-condition predicates. Old
  * values are used in pre-condition predicates. New and old values are
  * also used in predicate transformers. Intermediate values are used to
  * combine transformers.
  *
  * New values are directly referenced by the corresponding scalar
  * variable, a regular entity, with types integer, boolean, float,
  * complex or string. However they are printed with a different name to
  * avoid confusion between variables and values. The value name is the
  * variable name, suffixed by "#new".
  *
  * Old values and intermediate values are referenced via special
  * entities, distinguished by their names made of a specific prefix
  * followed by a number (see OLD_VALUE_PREFIX and
  * INTERMEDIATE_VALUE_PREFIX in transformer.h).
  *
  * The type of a value is the type of the associated entity. Equivalenced
  * variables are handled only if they all have the same type.
  *
  * Pre-values, a.k.a. old values, for local dynamic entities are
  * named "o#XX", where "XX" is a number. A pre-value is associated to
  * an entity via a hash table used by the function
  * entity_to_old_value(). Static aliasing (i.e.  EQUIVALENCEs) is
  * dealed with by associating the same value entity to the two
  * aliased variable entities.
  * 
  * Pre-values, a.k.a. old values, for analyzed scalar formal
  * parameters and analyzed scalar global variables are represented by
  * special entities named from the relevant variable entity, by
  * suffixing its name by "#init". These values have to be represented
  * by specific entities to let us consider parameter values at entry
  * point and to let us perform interprocedural semantics analysis.
  *
  * Intermediate values are referenced by special entities named "i#XX",
  * which are reused from procedure to procedure. Within a procedure,
  * each variable is linked to one intermediate value entity.
  *
  * These conventions are used by value_to_name(e) to generate EXTERNAL
  * names I#new, I#old and I#tmp and to print readable debug information.
  *
  * To accelerate conversions between new, old and intermediate value
  * entities, four hash tables must set up for each module before
  * semantics analysis is started or used:
  *
  *  - hash_entity_to_new_value associates an entity representing a new
  *  value to each scalar variable with an analyzed type; this value
  *  entity is most of the time the variable entity itself and thus this
  *  hash table almost represents an identity function; however,
  *  perfectly equivalenced variables are represented by one of them
  *  only and some eligible analyzable scalar variables are not mapped to
  *  anything because they are equivalenced to an array
  *
  *  - hash_entity_to_old_value associates an entity representing an old
  *  value to each scalar analyzable variable
  *
  *  - hash_entity_to_intermediate_value
  *
  *  - hash_value_to_name: associates user-readable names to value entity;
  *    used for debugging and display purposes.
  *
  * Francois Irigoin, December 1989 (updated June 2001,... August 2009)
  *
  * Modifications:
  *
  *  - only integer variables used to be analyzed; the analysis is
  *  extended to strings, bool and floating point scalar variables
  *  (Francois Irigoin, 14 June 2001).
  *
  *  - temporary values are added to deal with subexpressions and to
  *  analyze non-linear expressions (Francois Irigoin, 14 June 2001).
  *
  *  - the three mappings between variable entities and value entities
  *  were done differently for each transformer and based on the arguments
  *  field; efficiency was limited because the mappings were based on
  *  lists and because transformers had to be renamed before they could be
  *  combined; aliasis information had to be computed at each step instead
  *  of being factored out at hashing time (Francois Irigoin, April 90)
  *
  * - *** assigments to array or to non integer scalar can now affect
  *    transformer on integer scalar variables (Francois Irigoin, April
  *    90)*** no longer true - such integer scalar variables are just
  *    ignored; however, it would be easy to do slightly better (21 April
  *    1990)
  *
  *  - old value suffix is now "#init" because it's nicer in
  *  prettyprinting predicates; it's also worse for transformer but they
  *  are less interesting for the users (Francois Irigoin, 18 April 1990)
  *
  *  - fourth hash_table entity->new_value added to handle EQUIVALENCEs
  *  (Francois Irigoin, 18 April 1990)
  *
  *  - only analyzable scalar variables that are not aliased or that are
  *  aliased to another scalar variable with same type are analyzed; dubious
  *  interest but April 24th is close (Francois Irigoin, 21 April 1990)
  *
  * Bugs/Features:
  *
  *  - once the hash tables are lost, intraprocedural transformers are
  *    useless while they were still interpretable when mappings were
  *    based on arguments; they could be translated into the previous
  *    representation when their computation is completed (Francois Irigoin,
  *    April 1990)
  *
  * - hash tables are static; recursion is not possible for
  *    interprocedural analysis; topographical order will have to be used
  *    for it (Francois Irigoin, April 1990)
  *
  * Assumptions:
  *
  *  - all variable values and only variable values used in a module have
  *  names in value_to_name(); however, constants, which also are values,
  *  do not appear in this mapping;
  *
  *  - all variables whose values may be imported have entries in the
  *  three mappings entity_to_xxx_value() if they are defined within the
  *  current module, directly or indirectly thru calls; the information is
  *  derived interprocedurally from the memory effects which do not take
  *  aliasing into account; aliasing is handled using these three
  *  mappings; several exactly aliased variables are represented by the
  *  variable that appears first in the module effects, or by default, if
  *  none of these variables is in the scope of the current module, by any
  *  of them, probably the first one in the interprocedural effects;
  *  variables that are only read, i.e. used, are mapped only in
  *  entity_to_new_value();
  *
  *  - a summary_transformer imported for a callee may be difficult to
  *  print before it is translated in the current module frame;
  *
  *  - there is no absolute frame to express summary_preconditions for the
  *  time being (maybe, this has been fixed!)
  *
  * Francois Irigoin, 13 January 1994
  */

#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "effects.h"
#include "effects-util.h"
#include "constants.h"

#include "properties.h"

#include "misc.h"
#include "preprocessor.h"

#include "transformer.h"
#include "alias-classes.h"

/* STATIC VARIABLES */

/* Four global hash tables used to map scalar analyzable variable entities
 * onto the different kind of value entities and to map value entities
 * onto their external names.
 *
 * They prevent recursive calls.
 */
static hash_table hash_entity_to_new_value = hash_table_undefined;
static hash_table hash_entity_to_old_value = hash_table_undefined;
static hash_table hash_entity_to_intermediate_value = hash_table_undefined;
static hash_table hash_value_to_name = hash_table_undefined;
static hash_table hash_entity_to_user_value_name = hash_table_undefined;

static hash_table hash_reference_to_address_of_value = hash_table_undefined;

static hash_table hash_type_to_sizeof_value = hash_table_undefined;

bool hash_entity_to_values_undefined_p()
{
  return (hash_table_undefined_p(hash_entity_to_new_value));
}


/* Two counters used to assign meaningless value entities to local variables.
 * A special global prefix, SEMANTICS_MODULE_NAME, and two special local
 * prefixes, OLD_VALUE_PREFIX and INTERMEDIATE_VALUE_PREFIX, are also used
 */

static int local_intermediate_value_counter = 0;
static int local_old_value_counter = 0;
static int local_temporary_value_counter = 0;

void reset_value_counters()
{
  local_old_value_counter = 0;
  local_intermediate_value_counter = 0;
}

void reset_temporary_value_counter()
{
  local_temporary_value_counter = 0;
}

int number_of_temporary_values()
{
  return local_temporary_value_counter;
}

/* TYPING */

static bool analyze_integer_scalar_entities = true;
static bool analyze_boolean_scalar_entities = false;
static bool analyze_string_scalar_entities = false;
static bool analyze_float_scalar_entities = false;
static bool analyze_complex_scalar_entities = false;
static bool analyze_pointer_scalar_entities = false;
static bool analyze_constant_path = false;

void set_analyzed_types()
{
  analyze_integer_scalar_entities =
    get_bool_property("SEMANTICS_ANALYZE_SCALAR_INTEGER_VARIABLES");
  analyze_boolean_scalar_entities =
    get_bool_property("SEMANTICS_ANALYZE_SCALAR_BOOLEAN_VARIABLES");
  analyze_string_scalar_entities =
    get_bool_property("SEMANTICS_ANALYZE_SCALAR_STRING_VARIABLES");
  analyze_float_scalar_entities =
    get_bool_property("SEMANTICS_ANALYZE_SCALAR_FLOAT_VARIABLES");
  analyze_complex_scalar_entities =
    get_bool_property("SEMANTICS_ANALYZE_SCALAR_COMPLEX_VARIABLES");
  analyze_pointer_scalar_entities =
    get_bool_property("SEMANTICS_ANALYZE_SCALAR_POINTER_VARIABLES");
  analyze_constant_path =
    get_bool_property("SEMANTICS_ANALYZE_CONSTANT_PATH");
}

void reset_analyzed_types()
{
  analyze_integer_scalar_entities = true;
  analyze_boolean_scalar_entities = false;
  analyze_string_scalar_entities = false;
  analyze_float_scalar_entities = false;
  analyze_complex_scalar_entities = false;
  analyze_pointer_scalar_entities = false;
  analyze_constant_path = false;
}

bool integer_analyzed_p()
{
  return analyze_integer_scalar_entities;
}

bool boolean_analyzed_p()
{
  return analyze_boolean_scalar_entities;
}

bool string_analyzed_p()
{
  return analyze_string_scalar_entities;
}

bool float_analyzed_p()
{
  return analyze_float_scalar_entities;
}

bool complex_analyzed_p()
{
  return analyze_complex_scalar_entities;
}

bool pointer_analyzed_p()
{
  return analyze_pointer_scalar_entities;
}

bool constant_path_analyzed_p()
{
  return analyze_constant_path;
}


/* The entity is type of one of the analyzed types */
bool analyzable_basic_p(basic b)
{
  bool analyzable_p = false;

  if(analyze_integer_scalar_entities && basic_int_p(b))
    analyzable_p = true;
  else if(analyze_string_scalar_entities && basic_string_p(b))
    analyzable_p = true;
  else if(analyze_boolean_scalar_entities && basic_logical_p(b))
    analyzable_p = true;
  else if(analyze_float_scalar_entities && basic_float_p(b))
    analyzable_p = true;
  else if(analyze_pointer_scalar_entities && basic_pointer_p(b)) {
    pips_debug(9,"pointer type : %s\n", type_to_string(basic_pointer(b)));
    analyzable_p = true;
  }
  else if(analyze_complex_scalar_entities && basic_complex_p(b))
    analyzable_p = true;
  else if(analyze_integer_scalar_entities && basic_derived_p(b)) {
    entity de = basic_derived(b);
    type dt = entity_basic_concrete_type(de);
    analyzable_p = type_enum_p(dt);
  }
  else
    analyzable_p = false;

  return analyzable_p;
}

/* The entity is type of one of the analyzed types */
bool analyzable_type_p(type t)
{
  bool result = false;
  type bct = compute_basic_concrete_type(t);

  if(type_variable_p(bct)) {
    variable v = type_variable(bct);
    if (!volatile_variable_p(v)
        && ENDP(variable_dimensions(v))           //NL : for checking the dimension instead of entity_scalar_p which filter pointer type
    ) {
      result = analyzable_basic_p(variable_basic(v));
    }
  }

  return result;
}

/* The entity is type of one of the analyzed types */
bool analyzable_scalar_entity_p(entity e)
{
  bool result = false;

  if(!abstract_state_variable_p(e) && !typedef_entity_p(e) ) {
    type bct = entity_basic_concrete_type(e);
    result = analyzable_type_p(bct);
  }
  return result;
}

/* The constant may appear as a variable in the linear systems */
bool analyzed_constant_p(entity f)
{
  /* is f a 0-ary function, i.e. a constant of proper type? */
  if(entity_constant_p(f)) {
    basic b = variable_basic(type_variable
      (functional_result(type_functional(entity_type(f)))));

    /* integer and logical constants are handled explicitly */

    if(basic_string_p(b) && analyze_string_scalar_entities)
      return true;
    if(basic_float_p(b) && analyze_float_scalar_entities)
      return true;
    if(basic_complex_p(b) && analyze_complex_scalar_entities)
      return true;
    if(basic_pointer_p(b) && analyze_pointer_scalar_entities)
      return true;
  }
  return false;
}

bool analyzed_reference_p(reference r)
{
  pips_debug(7, "analyzed_reference_p, ref : %s\n", reference_to_string(r));
  bool result = false;

  // Check if the reference r is a constant path or not
  // TODO : strict_constant_path_p(r) return true when the ref is exactly a constant path
  // strict_constant_path_p always return false for the moment
  if (can_be_constant_path_p(r)) {
    pips_debug(9, "can_be_constant_path_p\n");
    if (strict_constant_path_p(r)) {
      pips_debug(9, "strict_constant_path_p\n");
      // if ref is a constant path
      if (analyze_constant_path) {
        bool to_be_free = false;
        type t = points_to_reference_to_type(r, &to_be_free);
        result = analyzable_type_p(t);
        if (to_be_free) {
          free_type(t);
        }
      }
    }
    else {
      pips_debug(9, "NO strict_constant_path_p\n");
      // if ref isn't a constant path
      result = true;
    }
  }
  else {
    pips_debug(9, "NO can_be_constant_path_p\n");
    // a[i]
    result = false;
  }

  pips_debug(7, "analyzed_reference_p result : %i\n", result);
  return result;
}



/* LOCAL VALUE ENTITY */

/* static entity make_local_value_entity(int n, bool old): find or generate
 * an entity representing an old or an intermediate value of number n
 */
static entity make_local_value_entity(int n, int nature, type t)
{
  entity v;
  /* 13 is a magic number that can accommodate 2 prefix characters,
     10 digits and a null character */
  char value_name[13];
  char * s;

  if(nature==0)
    (void) strcpy(value_name, OLD_VALUE_PREFIX);
  else if(nature==1)
    (void) strcpy(value_name, INTERMEDIATE_VALUE_PREFIX);
  else
    (void) strcpy(value_name, TEMPORARY_VALUE_PREFIX);
  (void) sprintf(value_name+2,"%d",n);
  pips_debug(8,"value name: %s\n",value_name);
  s = strdup(concatenate(SEMANTICS_MODULE_NAME,
			 MODULE_SEP_STRING, value_name, (char *) NULL));

  /* find entity or define it */
  v = gen_find_tabulated(s, entity_domain);
  if(v==entity_undefined)
    v = make_entity(s,
          copy_type(t),
          make_storage(is_storage_rom, UU),
          value_undefined);
  else {
    free(s);
    /* Another option might be to undefine types when counters are
       reset? */
    /* It is likely to take longer to compare types than to free one and
       allocate one... */
    /* Well, type_equal_p() replaces typedefs types by the underlying
       type which leads to disasters; see suppress_dead_code04.c */
    if(true || type_equal_p(entity_type(v), t)) {
      free_type(entity_type(v));
      entity_type(v) = copy_type(t);
    }
    else if(basic_string_p(variable_basic(type_variable(entity_type(v))))) {
      /* The previous test always returns true for strings */
      free_type(entity_type(v));
      entity_type(v) = copy_type(t);
    }
  }

  return v;
}

static entity make_local_old_value_entity(type t)
{
  return make_local_value_entity(local_old_value_counter++, 0, t);
}

static entity make_local_intermediate_value_entity(type t)
{
  return make_local_value_entity(local_intermediate_value_counter++, 1, t);
}

entity make_local_temporary_value_entity(type t)
{
  entity tv = entity_undefined;

  /* FI: is it easier to admit value of type void than to make a
     special case? Let see if it works... No, it's not a good idea
     because value are assumed of type variable in many places. */
  if(analyzable_type_p(t))
    tv = make_local_value_entity(local_temporary_value_counter++, 2, t);
  else
    pips_internal_error("Request for a temporary value with a non analyzable type");

  return tv;
}

entity make_local_temporary_value_entity_with_basic(basic b)
{
  type t = make_type(is_type_variable, make_variable(copy_basic(b), NIL,NIL));
  entity tmp = make_local_value_entity(local_temporary_value_counter++, 2, t);

  free_type(t);
  return tmp;
}

entity make_local_temporary_integer_value_entity()
{
  basic b = make_basic(is_basic_int, (void *) 4);
  type t = make_type(is_type_variable, make_variable(b, NIL,NIL));
  entity tmp = make_local_value_entity(local_temporary_value_counter++, 2, t);

  free_type(t);
  return tmp;
}

/* Return true if an entity is a local old value (such as "o#0" for a
   global value "i#init"...).
*/
bool local_old_value_entity_p(entity e)
{
  /* This is not a general test; it will only work for LOCAL values */
  return strncmp(entity_local_name(e), OLD_VALUE_PREFIX, 2) == 0;
}

bool local_intermediate_value_entity_p(entity e)
{
  /* this is not a general test; it will only work for LOCAL values */
  return strncmp(entity_local_name(e), INTERMEDIATE_VALUE_PREFIX, 2) == 0;
}

bool local_temporary_value_entity_p(entity e)
{
  /* this is not a general test; it will only work for LOCAL values */
  return strncmp(entity_local_name(e), TEMPORARY_VALUE_PREFIX, 2) == 0;
}

/* GLOBAL VALUES */

bool global_new_value_p(entity e)
{
  bool new = false;
  /* this is not a general test; it will only work for GLOBAL values */

  /* this function should always return false because new value = variable (FI) */

  /* => suf == NULL
     string suf = strchr(entity_local_name(e), SEMANTICS_SEPARATOR);

     pips_assert("global_new_value", suf != NULL);

     new = strcmp(entity_module_name(e), SEMANTICS_MODULE_NAME) != 0 &&
     strcmp(suf, NEW_VALUE_SUFFIX) == 0;
  */

  pips_assert("global_new_value", new == false && e==e);

  return new;
}

/* Return true if an entity is a global old value (such as
   "i#init"...). */
bool global_old_value_p(entity e)
{
  /* this is not a general test; it will only work for GLOBAL values */
  string suf = strchr(entity_local_name(e), SEMANTICS_SEPARATOR);
  bool old = false;

  if(suf!=NULL)
    old = strcmp(entity_module_name(e), SEMANTICS_MODULE_NAME) != 0 &&
      strcmp(suf, OLD_VALUE_SUFFIX) == 0;

  return old;
}

bool global_intermediate_value_p(entity e)
{
  /* this is not a general test; it will only work for GLOBAL values */
  string suf = strchr(entity_local_name(e), SEMANTICS_SEPARATOR);
  bool intermediate = false;

  if(suf!=NULL)
    intermediate = strcmp(entity_module_name(e), SEMANTICS_MODULE_NAME) != 0 &&
      strcmp(suf, INTERMEDIATE_VALUE_SUFFIX) == 0;

  return intermediate;
}

entity global_new_value_to_global_old_value(entity v_new)
{
  entity v_old = entity_undefined;

  /* There is no real test for global new values */

  pips_assert("new value must be a real variable entity, denoting the new value",
	      strcmp(entity_module_name(v_new), SEMANTICS_MODULE_NAME) != 0);

  v_old = (entity) gen_find_tabulated(concatenate(entity_name(v_new),
                                                  OLD_VALUE_SUFFIX,
                                                  NULL),
                                      entity_domain);
  if(v_old==NULL) v_old = entity_undefined;

  return v_old;
}

/* HASH TABLE USE
 *
 * Return a variable value name or map a variable to its different
 * variable values.
 *
 */

/* the '#' character used in value naming conflicts with the reserved
   character for struct naming*/
const char* global_value_name_to_user_name(const char* gn)
{
  const char* un = strrchr(gn, BLOCK_SEP_CHAR);

  if(un==NULL)
    un = local_name(gn);
  else
    un++;

  return un;
}

const char* external_value_name(entity e)
{
  entity m = get_current_module_entity();
  const char* s = hash_get(hash_value_to_name, (char *) e);

  if(s==HASH_UNDEFINED_VALUE && !variable_in_module_p(e,m))
    {
      if(global_new_value_p(e)) {
	entity a = entity_to_new_value(e);
	s = hash_get(hash_value_to_name, (char *) a);
      }
      else if(global_old_value_p(e)) {
	entity a = entity_to_old_value(e);
	s = hash_get(hash_value_to_name, (char *) a);
      }
      else if(global_intermediate_value_p(e)) {
	entity a = entity_to_intermediate_value(e);
	s = hash_get(hash_value_to_name, (char *) a);
      }
      else if(entity_constant_p(e)) {
          s = entity_name(e);
      }
      else if(null_pointer_value_entity_p(e)) {
          s = entity_name(e);
      }
      else {
	/* This should never occur. Please core dump! */
	pips_internal_error("\nUnexpected value \"%s\""
			    " for current module \"%s\"",
			    entity_name(e),
			    module_local_name(get_current_module_entity()));
      }
    }

  pips_assert("var must be bounded", s != HASH_UNDEFINED_VALUE);

  if(strcmp(module_local_name(m), module_name(s)) == 0
     ||strcmp(TOP_LEVEL_MODULE_NAME, module_name(s)) == 0) {
    //s = local_name(s);
    s = global_value_name_to_user_name(s);
  }

  return s;
}

/* This function is called many times when the constraints and the
   system of constraints are sorted using lexicographic information
   based on this particular value name. See for instance
   Semantics-New/freia_52.c. Hence it is memorized.
*/
const char * pips_user_value_name(entity e)
{
  pips_debug(9, "start with entity : %s\n", entity_name(e));
  const char* uvn = string_undefined;

  if(e == (entity) TCST) {
    uvn = "";
  }
  else {
    // To check the execution speed, uncomment the next line
    // return entity_name(e);
    uvn = hash_get(hash_entity_to_user_value_name, (char *) e);

    if(uvn==HASH_UNDEFINED_VALUE) {
      // Need to discriminate the case of an address_of value
      // because the hash table are reset between each pass
      if (address_of_value_entity_p(e)) {
        entity v = value_to_variable(e);
        string temp = strdup(entity_name(e));
        string indice = strstr(temp, "[");
        if (indice != NULL)
          *(indice+strlen(indice)-strlen(ADDRESS_OF_SUFFIX)) = '\0';
        uvn = strdup(concatenate("&", entity_user_name(v), indice, (char *) NULL));
        free(temp);
      }
      else if (null_pointer_value_entity_p(e)) {
        uvn = strdup("NULL");
      }
      else if (sizeof_value_entity_p(e)) {
        type t = entity_type(e);
        uvn = strdup(concatenate("sizeof(", type_to_full_string_definition(t), ")", (char *) NULL));
      }
      else {
        (void) gen_check((gen_chunk *) e, entity_domain);
        uvn = entity_has_values_p(e)? (string)entity_minimal_name(e) :
            external_value_name(e);
      }
        hash_put(hash_entity_to_user_value_name, (char *) e, uvn);
    }
  }
  pips_debug(9, "end with string : %s\n", uvn);
  return uvn;
}

entity entity_to_new_value(entity e)
{
  entity n;
  if((n = (entity) hash_get(hash_entity_to_new_value, (char *) e))
     == entity_undefined)
    pips_internal_error("unbounded entity %s",
			entity_name(e));
  return n;
}

entity entity_to_old_value(entity e)
{
  entity o;
  if((o = (entity) hash_get(hash_entity_to_old_value, (char *) e))
     == entity_undefined)
    pips_internal_error("unbounded entity %s",
			entity_name(e));
  return o;
}

entity entity_to_intermediate_value(entity e)
{
  entity i;
  if((i = (entity) hash_get(hash_entity_to_intermediate_value, (char *) e))
     == entity_undefined)
    pips_internal_error("unbounded entity %s",
			entity_name(e));
  return i;
}

entity reference_to_address_of_value(reference r)
{
  entity n;
  if((n = (entity) hash_get(hash_reference_to_address_of_value, (char *) r))
      == entity_undefined)
    pips_internal_error("unbounded reference %s",
        reference_to_string(r));
  return n;
}

entity type_to_sizeof_value(type t)
{
  entity i;
  if((i = (entity) hash_get(hash_type_to_sizeof_value, (char *) t))
      == entity_undefined)
    pips_internal_error("unbounded type %s : %s",
        type_to_string(t), type_to_full_string_definition(t));
  return i;
}

/* This function could be made more robust by checking the storage of
   e. Formal parameters of analyzable type always have values. */
bool entity_has_values_p(entity e)
{
  /* is e a variable whose value(s) (already) are analyzed?
   */
  pips_assert("value hash table is defined",
	      !hash_table_undefined_p(hash_entity_to_new_value));

  return hash_defined_p(hash_entity_to_new_value, (char *) e);
}

/* the following three functions are directly or indirectly relative
 * to the current module and its value hash tables.
 */

bool new_value_entity_p(entity e)
{
  /* since new values are always variable entities, hash_entity_to_new_value
     can be used for this test */

  pips_assert("new_value_entity_p",e != entity_undefined);

  return (entity) hash_get(hash_entity_to_new_value, (char *) e)
    == e;
}

bool old_value_entity_p(entity e)
{
  /* Temporary values do not have an external name. */
  /* OLD_VALUE_PREFIX is not used for global old values */
  /* string s = strstr(external_value_name(e), OLD_VALUE_SUFFIX); */
  /* string s = strstr(entity_local_name(e), OLD_VALUE_PREFIX); */

  if(!local_temporary_value_entity_p(e)) {
    string s1 = strstr(entity_local_name(e), OLD_VALUE_SUFFIX);
    // Need to remake the search for OLD_VALUE_PREFIX
    // bug for toto#...
    string s2 = strstr(entity_local_name(e), OLD_VALUE_PREFIX);
    // s2==entity_local_name(e) : for the case toto#...
    return s1!=NULL || (s2!=NULL && s2==entity_local_name(e));
  }
  else
    return false;
}

bool intermediate_value_entity_p(entity e)
{
  string s = strstr(external_value_name(e), INTERMEDIATE_VALUE_SUFFIX);

  return s!=NULL;
}

bool address_of_value_entity_p(entity e)
{
  string s = strstr(entity_local_name(e), ADDRESS_OF_SUFFIX);

  return s!=NULL;
}

bool sizeof_value_entity_p(entity e)
{
  string s = strstr(entity_local_name(e), SIZEOF_SUFFIX);

  return s!=NULL;
}

bool value_entity_p(entity e)
{
  /* tells if e is seen as a variable value in the current module */
  string s = hash_get(hash_value_to_name, (char *) e);

  if(s == (char*) HASH_UNDEFINED_VALUE) {
    return false;
  }
  else {
    return true;
  }
}

/* used with hash_table_fprintf */
static string string_identity(string s)
{ return s;}

void print_value_mappings()
{
  (void) fprintf(stderr,"\nhash table value to name:\n");
  hash_table_fprintf(stderr, (gen_string_func_t)dump_value_name, (gen_string_func_t)string_identity,
		     hash_value_to_name);

  (void) fprintf(stderr,"\nhash table entity to new value:\n");
  /*
    hash_table_fprintf(stderr, entity_local_name, external_value_name,
    hash_entity_to_new_value);
  */
  hash_table_fprintf(stderr, (gen_string_func_t)entity_minimal_name, (gen_string_func_t)entity_minimal_name,
		     hash_entity_to_new_value);

  (void) fprintf(stderr,"\nhash table entity to old value:\n");
  hash_table_fprintf(stderr, (gen_string_func_t)entity_minimal_name, (gen_string_func_t)external_value_name,
		     hash_entity_to_old_value);

  (void) fprintf(stderr, "\nhash table entity to intermediate value:\n");
  hash_table_fprintf(stderr, (gen_string_func_t)entity_minimal_name, (gen_string_func_t)external_value_name,
                         hash_entity_to_intermediate_value);

  (void) fprintf(stderr, "\nhash table reference to address_of value:\n");
  hash_table_fprintf(stderr, (gen_string_func_t)reference_to_string, (gen_string_func_t)external_value_name,
                         hash_reference_to_address_of_value);

  (void) fprintf(stderr, "\nhash table entity to sizeof value:\n");
  hash_table_fprintf(stderr, (gen_string_func_t)type_to_full_string_definition, (gen_string_func_t)external_value_name,
                         hash_type_to_sizeof_value);
}

static int mapping_to_value_number(hash_table h)
{
  size_t count = 0;
  list values = NIL;

  HASH_MAP(var, val, {
    if(!gen_in_list_p((entity) val, values)) {
      values = CONS(ENTITY,(entity) val, values);
      count++;
    }
  }, h);

  pips_assert("The number of insertions is equal to the list length",
	      count == gen_length(values));
  gen_free_list(values);
  return count;
}

/* Returns the list of entities in the mapping domain
 *
 * Could be more efficient to return a set.
 *
 * Note: this is a copy of mapping_to_value_number()
*/
static list mapping_to_domain_list(hash_table h)
{
  size_t count = 0;
  list values = NIL;

  HASH_MAP(var, val, {
    if(!gen_in_list_p((entity) val, values)) {
      values = CONS(ENTITY,(entity) var, values);
      count++;
    }
  }, h);

  pips_assert("The number of insertions is equal to the list length",
	      count == gen_length(values));

  return values;
}
#if 0
/* Returns the list of entities in the mapping range.
 *
 * Could be more efficient to return a set.
 *
 * Note: this is a copy of mapping_to_value_number()
*/
static list mapping_to_range_list(hash_table h)
{
  size_t count = 0;
  list values = NIL;

  HASH_MAP(var, val, {
    if(!gen_in_list_p((entity) val, values)) {
      values = CONS(ENTITY,(entity) val, values);
      count++;
    }
  }, h);

  pips_assert("The number of insertions is equal to the list length",
	      count == gen_length(values));

  return values;
}
#endif

/* Return the list of all analyzed variables which are modified in
   the current module. If they are modified, they must have old
   values. */
list modified_variables_with_values()
{
  /* The intermediate values could be used as well */
  /*
  list ivl = mapping_to_domain_list(hash_entity_to_old_value); // initial
							       // value list
  list wvl = NIL; // written variable list

  FOREACH(ENTITY, e, ivl) {
    entity wv = new_value_to_variable(e);
    ivl = CONS(ENTITY, wv, ivl);
  }

  gen_reverse(wvl);
  */
  list wvl = mapping_to_domain_list(hash_entity_to_old_value);
  return wvl;
}

void test_mapping_entry_consistency()
{
  int nbo = 0;
  int nbi = 0;
  int nbn = 0;

  pips_assert("The number of old values is equal to the number of intermediate values",
	      ((nbo= hash_table_entry_count(hash_entity_to_old_value))
	       == (nbi= hash_table_entry_count(hash_entity_to_intermediate_value))));
  /* This second assert is too strong when some analyzable variables are
     equivalenced together. The number of values required is smaller
     since two (or more) different variables share the same value: the
     number of entries in the tables is greater than the number of
     values. We must compute the number of values in each table.
  */
  /*
    pips_assert("The number of values is greater than the number"
    " of new, old and intermediate values",
    hash_table_entry_count(hash_value_to_name) >=
    hash_table_entry_count(hash_entity_to_new_value)
    + nbo + nbi);
  */
  nbo = mapping_to_value_number(hash_entity_to_old_value);
  nbi = mapping_to_value_number(hash_entity_to_intermediate_value);
  nbn = mapping_to_value_number(hash_entity_to_new_value);

  /* Why greater instead of equal? Because the equivalence variable
     appears in the equivalence equations although it should never
     appear in regular constraints, except under its canonical name. */
  pips_assert("The number of values with a name is greater than the number"
	      " of new, old and intermediate values",
	      hash_table_entry_count(hash_value_to_name) >=
	      nbn + nbo + nbi);
}

int number_of_analyzed_values()
{
  return hash_table_entry_count(hash_value_to_name);
}

int aproximate_number_of_analyzed_variables()
{
  /* FI: I do not know if equivalenced variables are well taken into account*/
  return hash_table_entry_count(hash_entity_to_new_value);
}

/* FI: looks more like the number of values used. */
int number_of_analyzed_variables()
{
  return hash_table_entry_count(hash_value_to_name);
}

void allocate_value_mappings(int n, int o, int i)
{
  pips_assert("undefined mappings for allocation",
      hash_table_undefined_p(hash_entity_to_new_value) &&
      hash_table_undefined_p(hash_entity_to_old_value) &&
      hash_table_undefined_p(hash_entity_to_intermediate_value) &&
      hash_table_undefined_p(hash_reference_to_address_of_value) &&
      hash_table_undefined_p(hash_type_to_sizeof_value) &&
      hash_table_undefined_p(hash_value_to_name));

  /* hash_warn_on_redefinition(); */
  hash_entity_to_new_value = hash_table_make(hash_pointer, n);
  hash_entity_to_old_value = hash_table_make(hash_pointer, o);
  hash_entity_to_intermediate_value =
    hash_table_make(hash_pointer, i);
  hash_value_to_name =
    hash_table_make(hash_pointer, n + o + i + n);           //The last +n for the pointer
  hash_entity_to_user_value_name =
    hash_table_make(hash_pointer, n + o + i + n);           //The last +n for the pointer
  hash_reference_to_address_of_value = hash_table_make(hash_pointer, n);
  hash_type_to_sizeof_value = hash_table_make(hash_pointer, n);
}

static void reset_value_mappings(void)
{
  hash_entity_to_new_value = hash_table_undefined;
  hash_entity_to_old_value = hash_table_undefined;
  hash_entity_to_intermediate_value = hash_table_undefined;
  hash_value_to_name = hash_table_undefined;
  hash_entity_to_user_value_name = hash_table_undefined;
  hash_reference_to_address_of_value = hash_table_undefined;
  hash_type_to_sizeof_value = hash_table_undefined;
}

bool hash_value_to_name_undefined_p()
{ return hash_table_undefined_p(hash_value_to_name);
}

/* To be called by error handler only. Potential memory leak. */
void error_reset_value_mappings(void)
{
  reset_value_mappings();
}

/* Normal call to free the mappings */

void free_value_mappings(void)
{
  pips_assert("no free of undefined mappings",
	      !hash_table_undefined_p(hash_entity_to_old_value));
  error_free_value_mappings();
}

/* To be called by an error handler */

void error_free_value_mappings(void)
{
  /* free previous hash tables, desallocate names; this implies ALL
     value names were malloced and were not pointer to a ri part */

  /* the three tables are assumed to be allocated all together */

  /* free names in hash_value_to_name; the other two hash tables
     contain pointers to the entity tabulated domain and thus need
     no value freeing */
  /* k is discovered unused by lint; it is syntaxically necessary */
  HASH_MAP(k, v, {free(v);}, hash_value_to_name);
  // Do not deallocate the names: they are pointers towards parts of
  // entity names
  //HASH_MAP(k, v, {free(v);}, hash_entity_to_user_value_name);
  /* free the three tables themselves */
  hash_table_free(hash_entity_to_new_value);
  hash_table_free(hash_entity_to_old_value);
  hash_table_free(hash_entity_to_intermediate_value);
  hash_table_free(hash_value_to_name);
  hash_table_free(hash_entity_to_user_value_name);
  hash_table_free(hash_reference_to_address_of_value);
  hash_table_free(hash_type_to_sizeof_value);

  reset_value_mappings();
  reset_temporary_value_counter();
  reset_analyzed_types();
}

/* HASH TABLE INITIALIZATION */

/* void add_new_value_name(entity e): add a new value name for entity e */
static void add_new_value_name(entity e)
{
  string new_value_name =
    strdup(concatenate(entity_name(e), NEW_VALUE_SUFFIX,
		       (char *) NULL));
  pips_debug(8,"begin: for %s\n", entity_name(e));
  if(hash_get(hash_value_to_name, (char *) e) == HASH_UNDEFINED_VALUE)
    hash_put(hash_value_to_name, (char *) e, (char *) new_value_name);
  else
    free(new_value_name);
}

void add_address_of_value(reference r, type t)
{
  entity e = reference_variable(r);
  entity address_of_value;
  string address_of_value_name;
  string indice = strstr(reference_to_string(r), "[");

  if (indice != NULL)
  address_of_value_name = concatenate(
      entity_name(e), indice, ADDRESS_OF_SUFFIX, (char *) NULL);
  else
    address_of_value_name = concatenate(
        entity_name(e), ADDRESS_OF_SUFFIX, (char *) NULL);


  //address_of_value = gen_find_entity(address_of_value_name);
  address_of_value = gen_find_tabulated(address_of_value_name, entity_domain);
  //if (entity_undefined_p(address_of_value))
  if(address_of_value == entity_undefined)
    address_of_value = make_entity(strdup(address_of_value_name),
                                  copy_type(t),
                                  make_storage(is_storage_rom, UU),
                                  value_undefined);

  /* add the couple (e, address_of_value) */
  if(hash_get(hash_reference_to_address_of_value, (char *) r) == HASH_UNDEFINED_VALUE) {
    hash_put(hash_reference_to_address_of_value, (char *) r, (char *) address_of_value);
    /* add its name */
    hash_put(hash_value_to_name, (char *) address_of_value,
               strdup(entity_name(address_of_value)));
    // The next table be reset after the analysis, can't be reuse for the prettyprint, so useless to add in the table?
    entity v = value_to_variable(address_of_value);
    hash_put(hash_entity_to_user_value_name, (char *) address_of_value,
               strdup(concatenate("&", entity_user_name(v), indice, (char *) NULL)));
  }
}

void add_sizeof_value(type t)
{
  entity sizeof_value;
  string sizeof_value_name;

  // improvment create the entity with the name of type
  // issue because of value_to_variable
  /* find the sizeof entity if possible, else, generate it */
  sizeof_value_name =
      concatenate(
          TOP_LEVEL_MODULE_NAME, MODULE_SEP_STRING,
          type_to_full_string_definition(t), SIZEOF_SUFFIX, (char *) NULL);
  sizeof_value = gen_find_tabulated(sizeof_value_name, entity_domain);
  if(sizeof_value == entity_undefined)
    sizeof_value =
        make_entity(strdup(sizeof_value_name),
            //make_type(is_type_variable,
            //    make_variable(make_basic(is_basic_int, (void*)sizeof(int)),NIL,NIL)),
            copy_type(t),
            make_storage(is_storage_rom, UU),
            value_undefined);
  /* add the couple (e, old_value) */
  if(hash_get(hash_type_to_sizeof_value, (char *) t) == HASH_UNDEFINED_VALUE) {
    hash_put(hash_type_to_sizeof_value, (char *) t, (char *) sizeof_value);
    /* add its name */
    hash_put(hash_value_to_name, (char *) sizeof_value,
        strdup(entity_name(sizeof_value)));
    hash_put(hash_entity_to_new_value, (char *) sizeof_value,
        (char *) sizeof_value);
    hash_put(hash_entity_to_user_value_name, (char *) sizeof_value,
        concatenate("sizeof(", type_to_full_string_definition(t), ")", (char *) NULL));
  }
}

void add_new_value(entity e)
{
  pips_debug(8,"begin: for %s\n", entity_name(e));
  if(hash_get(hash_entity_to_new_value, (char *) e)
     == HASH_UNDEFINED_VALUE) {
    hash_put(hash_entity_to_new_value, (char *) e, (char *) e);
    add_new_value_name(e);
  }
}

void add_new_alias_value(entity e, entity a)
{
  entity v = entity_undefined;
  pips_debug(8, "begin: for %s and %s\n",
	entity_name(e),entity_name(a));
  pips_assert("hash_entity_to_new_value is defined",
	      (hash_get(hash_entity_to_new_value, (char *) e)
	       == HASH_UNDEFINED_VALUE));

  v = entity_to_new_value(a);

  hash_put(hash_entity_to_new_value, (char *) e, (char *) v);
  /* add_new_value_name(e); */
}

entity external_entity_to_new_value(entity e)
{
  /* there is no decisive test here; a necessary condition is used instead */

  entity e_new;

  pips_assert("e must be a type analyzable scalar variable", analyzable_scalar_entity_p(e));
  e_new = e;
  return e_new;
}

entity external_entity_to_old_value(entity e)
{
  /* find the old value entity if possible or abort
     I should never have to call this function on an local entity
     (local entities include static local variables) */

  entity old_value;
  string old_value_name;

  old_value_name = concatenate(entity_name(e),OLD_VALUE_SUFFIX,
			       (char *) NULL);
  old_value = gen_find_tabulated(old_value_name, entity_domain);

  pips_assert("external_entity_to_old_value", old_value!=entity_undefined);

  return old_value;
}

void add_old_value(entity e)
{
  entity old_value;
  string old_value_name;

  /* find the old value entity if possible, else, generate it */
  old_value_name = concatenate(entity_name(e),OLD_VALUE_SUFFIX,
			       (char *) NULL);
  old_value = gen_find_tabulated(old_value_name, entity_domain);
  if(old_value == entity_undefined)
    old_value = make_entity(strdup(old_value_name),
			    copy_type(entity_type(e)),
			    make_storage(is_storage_rom, UU),
			    value_undefined);
  /* add the couple (e, old_value) */
  if(hash_get(hash_entity_to_old_value, (char *) e) == HASH_UNDEFINED_VALUE) {
    hash_put(hash_entity_to_old_value, (char *) e, (char *) old_value);
    /* add its name */
    hash_put(hash_value_to_name, (char *) old_value,
	     strdup(entity_name(old_value)));
  }
}

void add_old_alias_value(entity e, entity a)
{
  entity v = entity_undefined;

  pips_assert("add_old_alias_valued", (hash_get(hash_entity_to_old_value, (char *) e)
				       == HASH_UNDEFINED_VALUE));

  v = entity_to_old_value(a);

  hash_put(hash_entity_to_old_value, (char *) e, (char *) v);
  /* add_new_value_name(e); */
}

void add_intermediate_value(entity e)
{
  entity intermediate_value;

  /* get a new intermediate value, if necessary */
  if((intermediate_value =
      (entity) hash_get(hash_entity_to_intermediate_value, (char *) e))
     == (entity) HASH_UNDEFINED_VALUE) {
    intermediate_value = make_local_intermediate_value_entity(entity_type(e));
    hash_put(hash_entity_to_intermediate_value, (char *) e,
	     (char *) intermediate_value);
    /* add its (external) name */
    hash_put(hash_value_to_name, (char *) intermediate_value,
	     strdup(concatenate(entity_name(e),
				INTERMEDIATE_VALUE_SUFFIX,
				(char *) NULL)));
  }
}

void add_intermediate_alias_value(entity e, entity a)
{
  entity v = entity_undefined;

  pips_assert("add_intermediate_alias_valued",
	      (hash_get(hash_entity_to_intermediate_value, (char *) e)
	       == HASH_UNDEFINED_VALUE));

  v = entity_to_intermediate_value(a);

  hash_put(hash_entity_to_intermediate_value, (char *) e, (char *) v);
  /* add_new_value_name(e); */
}

void add_local_old_value(entity e)
{
  entity old_value;

  /* get a new old value, if necessary */
  if((old_value =
      (entity) hash_get(hash_entity_to_old_value, (char *) e))
     == (entity) HASH_UNDEFINED_VALUE) {
    old_value = make_local_old_value_entity(entity_type(e));
    hash_put(hash_entity_to_old_value, (char *) e, (char *) old_value);
    /* add its (external) name */
    hash_put(hash_value_to_name, (char *) old_value,
	     strdup(concatenate(entity_name(e), OLD_VALUE_SUFFIX,
				(char *) NULL)));
  }
}

void add_local_intermediate_value(entity e)
{
  entity intermediate_value;

  /* get a new intermediate value, if necessary */
  if((intermediate_value =
      (entity) hash_get(hash_entity_to_intermediate_value, (char *) e))
     == (entity) HASH_UNDEFINED_VALUE) {
    intermediate_value = make_local_intermediate_value_entity(entity_type(e));
    hash_put(hash_entity_to_intermediate_value, (char *) e,
	     (char *) intermediate_value);
    /* add its (external) name */
    hash_put(hash_value_to_name, (char *) intermediate_value,
	     strdup(concatenate(entity_name(e),
				INTERMEDIATE_VALUE_SUFFIX,
				(char *) NULL)));
  }
}

void remove_entity_values(entity e, bool readonly)
{
  entity new_value = entity_to_new_value(e);
  const char* s;

  /* pips_assert("remove_entity_values", e != entity_undefined); */
  pips_assert("remove_entity_values", new_value != entity_undefined);

  s = external_value_name(new_value);
  pips_assert("remove_entity_values", s != (char *) NULL);
  (void) hash_del(hash_value_to_name, (char *) new_value);
  (void) hash_del(hash_entity_to_new_value, (char *) e);

  if(!readonly) {
    entity old_value = entity_to_old_value(e);
    entity intermediate_value = entity_to_intermediate_value(e);

    pips_assert("remove_entity_values", old_value != entity_undefined);
    pips_assert("remove_entity_values",
		intermediate_value!=entity_undefined);

    s = external_value_name(old_value);
    pips_assert("remove_entity_values", s != (char *) NULL);
    (void) hash_del(hash_value_to_name, (char *) old_value);
    s = external_value_name(intermediate_value);
    pips_assert("remove_entity_values", s != (char *) NULL);
    (void) hash_del(hash_value_to_name, (char *) intermediate_value);

    (void) hash_del(hash_entity_to_old_value, (char *) e);
    (void) hash_del(hash_entity_to_intermediate_value, (char *) e);
  }
}

void add_synonym_values(entity e, entity eq, bool readonly)
{
  /* e and eq are entities whose values are always equal because they
   * share the exact same memory location (i.e. they are alias). Values
   * for e have already been declared. Values for eq have to be
   * declared. */
  entity new_value = entity_to_new_value(e);
  entity intermediate_value;

  pips_debug(8, "Begin for registered variable %s"
	     " equivalenced with new variable %s"
	     " with status %s\n",
	     entity_local_name(e), entity_local_name(eq),
	     readonly? "readonly" : "read/write");

  hash_put(hash_entity_to_new_value, (char *) eq,
	   (char *) new_value);
  if(!readonly) {
    entity old_value = entity_to_old_value(e);
    intermediate_value = entity_to_intermediate_value(e);
    add_old_value(eq);
    hash_put(hash_entity_to_intermediate_value, (char *) eq,
	     (char *) intermediate_value);
    hash_update(hash_entity_to_old_value, (char *) eq, 
		(char *) old_value);
    /* hash_put(hash_value_to_name, (char *) old_value_eq, entity_name(old_value_eq)); */
  }
  /* The name does not change. It is not used until the equivalence
     equations are added */
  hash_put(hash_value_to_name, (char *) eq,
	   strdup(concatenate(entity_name(eq),
			      NEW_VALUE_SUFFIX, (char *) NULL)));

  pips_debug(8, "End\n");
}


/* Get the primitive variable associated to any value involved in a
   transformer.

   For example can associate values such as "o#0" to "i" (via "i#init").

   This function used to be restricted to values seen by the current
   module. It was extended to values in general to cope with translation
   issues.
*/
entity value_to_variable(entity val)
{
  entity var = entity_undefined;
  int l_suffix = -1; /* for gcc only! */
  string s = hash_get(hash_value_to_name, (char *) val);
  string var_name;

  /* pips_assert("value_to_variable", s != HASH_UNDEFINED_VALUE); */

  /* pips_assert("value_to_variable",
     strchr(entity_name(val), (int) SEMANTICS_SEPARATOR) != NULL); */

  // TODO : Maybe redesign the search of suffix/prefix
  //        Can bug if the variable name is/begin/end with
  //        o, i, t, new, init, int, address_of see SUFFIX and PREFIX in transformer.h
  if(s == HASH_UNDEFINED_VALUE) {
    /* this may be a value, but it is unknown in the current module */
    string val_name = entity_name(val);

    if(strstr(val_name, NEW_VALUE_SUFFIX) != NULL)
      l_suffix = strlen(NEW_VALUE_SUFFIX);
    else if(strstr(val_name, OLD_VALUE_SUFFIX) != NULL)
      l_suffix = strlen(OLD_VALUE_SUFFIX);
    else if(strstr(val_name, INTERMEDIATE_VALUE_SUFFIX) != NULL)
      l_suffix = strlen(INTERMEDIATE_VALUE_SUFFIX);
    else if(strstr(val_name, ADDRESS_OF_SUFFIX) != NULL)
      l_suffix = strlen(ADDRESS_OF_SUFFIX);
    else if(strstr(val_name, SIZEOF_SUFFIX) != NULL)
      l_suffix = 0;
    else if(strchr(val_name, (int) SEMANTICS_SEPARATOR) == NULL) {
      /* new values in fact have no suffixes... */
      l_suffix = 0;
    }
    else
      pips_internal_error("%s is not a non-local value", entity_name(val));

    s = val_name;
  }
  else {
    if(sizeof_value_entity_p(val))
      l_suffix = 0;
    else if(new_value_entity_p(val))
      l_suffix = strlen(NEW_VALUE_SUFFIX);
    else if(old_value_entity_p(val))
      l_suffix = strlen(OLD_VALUE_SUFFIX);
    else if(intermediate_value_entity_p(val))
      l_suffix = strlen(INTERMEDIATE_VALUE_SUFFIX);
    else if(address_of_value_entity_p(val))
      l_suffix = strlen(ADDRESS_OF_SUFFIX);
    else
      /* It can be an equivalenced variable... Additional testing
	 should be performed! */
      pips_internal_error("%s is not a locally visible value",
			  entity_name(val));
  }

  var_name = strdup(s);
  *(var_name+strlen(var_name)-l_suffix) = '\0';
  if(address_of_value_entity_p(val)) {
    string temp = strstr(var_name, "[");
    if(temp != NULL)
      *temp = '\0';
  }

  var = gen_find_tabulated(var_name, entity_domain);
  free(var_name);

  if( var == entity_undefined )
    pips_internal_error("no related variable for val=%s",
			entity_name(val));

  return var;
}

entity old_value_to_new_value(entity o_val)
{
  entity var = entity_undefined;
  entity n_val = entity_undefined;

  var = value_to_variable(o_val);
  /* o_val = variable_to_old_value(var); */
  n_val = entity_to_new_value(var);

  return n_val;
}

entity new_value_to_old_value(entity n_val)
{
  entity o_val = entity_undefined;

  /* FI: correct code
     entity var = entity_undefined;

     var = value_to_variable(n_val);
     o_val = entity_to_old_value(var);
  */

  /* FI: faster code */
  o_val = entity_to_old_value(n_val);

  return o_val;
}

/* Static aliasing. useful for Fortran, useless for C. When no alias
   is found, an undefined entity must be returned. */
entity value_alias(entity e)
{
  entity a = entity_undefined;

  pips_debug(8,"begin: for %s\n", entity_name(e));

  /* lookup the current value name mapping and return an arbitraty "representant"
     of the interprocedural alias set of e; the equivalence relation is "has
     same location" */
  HASH_MAP(var, val, {
    if(variable_entity_p((entity) var) && entities_may_conflict_p((entity) var, e)) {
      a = (entity) var;
      break;
    }
  }, hash_value_to_name);

  if(a==entity_undefined)
    pips_debug(8, "return: %s\n", "entity_undefined");
  else
    pips_debug(8, "return: %s\n", entity_name(a));

  return a;
}

/* for debugging purposes */
string value_full_name(entity v)
{
  return entity_name(v);
}

/* For debugging purposes, we might have to print system with temporary
   values */
const char* readable_value_name(entity v)
{
  const char* n ;

  if(local_temporary_value_entity_p(v)) {
    n = entity_local_name(v);
  }
  else {
    n = external_value_name(v);
  }

  return n;
}

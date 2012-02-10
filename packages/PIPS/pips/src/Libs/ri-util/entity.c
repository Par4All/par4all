/*

  $Id$

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
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
/* Functions closely related to the entity class, constructors, predicates,...
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "linear.h"
#include "constants.h"

#include "genC.h"
#include "newgen_set.h"
#include "misc.h"
#include "ri.h"
#include "properties.h"
#include "pipsdbm.h"

#include "ri-util.h"

/********************************************************************/
/* Static variable to memoïze some entities for performance reasons */
/********************************************************************/


/* variables to store internal entities created at bootstrap
   beware: these variables are intialized during the parser phase
*/
static bool internal_static_entities_initialized_p = false;
/* effects package entities */
static entity rand_gen_ent  = entity_undefined;
static entity malloc_effect_ent  = entity_undefined;
static entity memmove_effect_ent  = entity_undefined;
static entity time_effect_ent  = entity_undefined;
static entity luns_ent  = entity_undefined;
static entity io_ptr_ent  = entity_undefined;
static entity io_eof_ent  = entity_undefined;
static entity io_error_luns_ent  = entity_undefined;

/* continue statement */
static entity continue_ent = entity_undefined;

/* variables to store entities from standard includes */
/* As they are not created at bootstrap as the internal entities,
   they cannot be initialized at the same time because the
   corresponding entities may not have been already created
*/
static bool std_static_entities_initialized_p = false;
/* stdio files entities */
static entity stdin_ent = entity_undefined;
static entity stdout_ent = entity_undefined;
static entity stderr_ent = entity_undefined;

#define STATIC_ENTITY_CACHE_SIZE 128
static entity *static_entity_cache[STATIC_ENTITY_CACHE_SIZE];
static size_t static_entity_size=0;

/* beware: cannot be called on creating the database */
void set_std_static_entities()
{
  if (!std_static_entities_initialized_p)
    {
      stdin_ent = FindEntity(TOP_LEVEL_MODULE_NAME, "stdin");
      stdout_ent = FindEntity(TOP_LEVEL_MODULE_NAME, "stdout");
      stderr_ent = FindEntity(TOP_LEVEL_MODULE_NAME, "stderr");
      std_static_entities_initialized_p = true;
    }
}

/* beware: cannot be called on creating the database */
void set_internal_static_entities()
{
  if (!internal_static_entities_initialized_p)
    {
      rand_gen_ent  = FindOrCreateEntity(RAND_EFFECTS_PACKAGE_NAME,
					 RAND_GEN_EFFECTS_NAME);
      malloc_effect_ent  = FindOrCreateEntity(MALLOC_EFFECTS_PACKAGE_NAME,
				       MALLOC_EFFECTS_NAME);
      memmove_effect_ent  = FindOrCreateEntity(MEMMOVE_EFFECTS_PACKAGE_NAME,
					MEMMOVE_EFFECTS_NAME);
      time_effect_ent  = FindOrCreateEntity(TIME_EFFECTS_PACKAGE_NAME,
					TIME_EFFECTS_VARIABLE_NAME);
      luns_ent  = FindOrCreateEntity(IO_EFFECTS_PACKAGE_NAME,
				     IO_EFFECTS_ARRAY_NAME);
      io_ptr_ent  = FindOrCreateEntity(IO_EFFECTS_PACKAGE_NAME,
				       IO_EFFECTS_PTR_NAME);
      io_eof_ent  = FindOrCreateEntity(IO_EFFECTS_PACKAGE_NAME,
				       IO_EOF_ARRAY_NAME);
      io_error_luns_ent  = FindOrCreateEntity(IO_EFFECTS_PACKAGE_NAME,
					      IO_ERROR_ARRAY_NAME);

      continue_ent = FindOrCreateTopLevelEntity(CONTINUE_FUNCTION_NAME);

      internal_static_entities_initialized_p = true;
    }
}

void reset_internal_static_entities()
{

  rand_gen_ent  = entity_undefined;
  malloc_effect_ent  = entity_undefined;
  memmove_effect_ent  = entity_undefined;
  time_effect_ent  = entity_undefined;
  luns_ent  = entity_undefined;
  io_ptr_ent  = entity_undefined;
  io_eof_ent  = entity_undefined;
  io_error_luns_ent  = entity_undefined;

  continue_ent = entity_undefined;
  for(size_t i =0;i< static_entity_size;i++)
      *static_entity_cache[i]=entity_undefined;
  static_entity_size=0;
  internal_static_entities_initialized_p = false;
}

void reset_std_static_entities()
{
  stdin_ent = entity_undefined;
  stdout_ent = entity_undefined;
  stderr_ent = entity_undefined;
  std_static_entities_initialized_p = false;
}

void reset_static_entities()
{
  reset_std_static_entities();
  reset_internal_static_entities();
}


/* add given entity to the set of entities that must reset upon workspace deletion
 * practically, all static entities should be stored that way
 */
void register_static_entity(entity *e) {
    static_entity_cache[static_entity_size++]=e;
    pips_assert("static entity cache is large enough",static_entity_size<STATIC_ENTITY_CACHE_SIZE);
}


/********************************************************************/

static set io_functions_set = set_undefined;
static set arithmetic_functions_set = set_undefined;

void print_entities(list l)
{
  FOREACH(ENTITY, e, l) {
    fprintf(stderr, "%s ", entity_name(e));
  }
}

void print_entity_set(set s)
{
  /* For some reason, here entity is not capitalized. */
  SET_FOREACH(entity, e, s) {
    fprintf(stderr, "%s ", entity_name(e));
  }
}


static void print_dimension(dimension d)
{
    fprintf(stderr,"dimension :\n");
    print_expression(dimension_lower(d));
    print_expression(dimension_upper(d));
}
/* print_entity_variable(e)
 *
 * if it is just a variable, the type is printed,
 * otherwise just the entity name is printed
 */
void print_entity_variable(entity e)
{
    variable v;

    (void) fprintf(stderr,"name: %s\n",entity_name(e));

    if (!type_variable_p(entity_type(e)))
	return;

    v = type_variable(entity_type(e));

    fprintf(stderr,"basic %s\n",basic_to_string(variable_basic(v)));
    gen_map((gen_iter_func_t)print_dimension, variable_dimensions(v));
}

bool unbounded_expression_p(expression e)
{
  syntax s = expression_syntax(e);
  if (syntax_call_p(s))
    {
      const char* n = entity_local_name(call_function(syntax_call(s)));
      if (same_string_p(n, UNBOUNDED_DIMENSION_NAME))
	return true;
    }
  return false;
}

expression make_unbounded_expression()
{
  return MakeNullaryCall(CreateIntrinsic(UNBOUNDED_DIMENSION_NAME));
}

/* FI: this piece of code must have been duplicated somewhere else in
   an effect library */
list make_unbounded_subscripts(int d)
{
  list sl = NIL;
  int i;

  for(i=0; i<d; i++) {
    sl = CONS(EXPRESSION, make_unbounded_expression(), sl);
  }

  return sl;
}


/* The source language is not specified. Might not work with C because
   of module_local_name. Also, the compilation unit is undefined.


   It might be necessary to declare the four areas in
   code_declarations.

   See also InitAreas() and init_c_areas(), which use global variables
   and hence cannot be used outside of the parsers. Beware of the
   consistency between this function and those two.

   See also MakeCurrentFunction(), which is part of the Fortran
   parser.
 */
entity make_empty_module(const char* full_name,
				type r, language l)
{
  const char* name;
  entity e = gen_find_tabulated(full_name, entity_domain);
  entity DynamicArea, StaticArea, StackArea, HeapArea;

  /* FC: added to allow reintrance in HPFC */
  if (e!=entity_undefined)
    {
      pips_debug(1,"module %s already exists, returning it\n", full_name);
      return e;
    }

  pips_assert("undefined", e == entity_undefined);

  e = make_entity
    (strdup(full_name),
     make_type_functional(
	       make_functional(NIL, r)),
     make_storage_rom(),
     make_value_code(
		make_code(NIL, strdup(""), make_sequence(NIL),NIL,
			  l)));

  name = module_local_name(e);
  DynamicArea = FindOrCreateEntity(name, DYNAMIC_AREA_LOCAL_NAME);
  entity_type(DynamicArea) = make_type_area(make_area(0, NIL));
  entity_storage(DynamicArea) = make_storage_rom();
  entity_initial(DynamicArea) = make_value_unknown();
  entity_kind(DynamicArea) = ABSTRACT_LOCATION | ENTITY_DYNAMIC_AREA ;
  AddEntityToDeclarations(DynamicArea, e);

  StaticArea = FindOrCreateEntity(name, STATIC_AREA_LOCAL_NAME);
  entity_type(StaticArea) = make_type_area(make_area(0, NIL));
  entity_storage(StaticArea) = make_storage_rom();
  entity_initial(StaticArea) = make_value_unknown();
  entity_kind(StaticArea) = ABSTRACT_LOCATION | ENTITY_STATIC_AREA ;
  AddEntityToDeclarations(StaticArea, e);

  StackArea = FindOrCreateEntity(name, STACK_AREA_LOCAL_NAME);
  entity_type(StackArea) = make_type_area(make_area(0, NIL));
  entity_storage(StackArea) = make_storage_rom();
  entity_initial(StackArea) = make_value_unknown();
  entity_kind(StackArea) = ABSTRACT_LOCATION | ENTITY_STACK_AREA ;
  AddEntityToDeclarations(StackArea, e);

  HeapArea = FindOrCreateEntity(name, HEAP_AREA_LOCAL_NAME);
  entity_type(HeapArea) = make_type_area(make_area(0, NIL));
  entity_storage(HeapArea) = make_storage_rom();
  entity_initial(HeapArea) = make_value_unknown();
  entity_kind(HeapArea) = ABSTRACT_LOCATION | ENTITY_HEAP_AREA ;
  AddEntityToDeclarations(HeapArea, e);

  return(e);
}

entity make_empty_program(const char* name,language l)
{
  string full_name = concatenate(TOP_LEVEL_MODULE_NAME
				 MODULE_SEP_STRING  MAIN_PREFIX, name, NULL);
  return make_empty_module(full_name, make_type_void(NIL),l);
}

entity make_empty_subroutine(const char* name,language l)
{
  string full_name = concatenate(TOP_LEVEL_MODULE_NAME
				 MODULE_SEP_STRING, name, NULL);
  return make_empty_module(full_name, make_type_void(NIL),l);
}

entity make_empty_f95module(const char* name,language l)
{
  pips_assert("Module are only defined in Fortran95",language_fortran95_p(l));
  string full_name = concatenate(TOP_LEVEL_MODULE_NAME
         MODULE_SEP_STRING, F95MODULE_PREFIX, name, NULL);
  return make_empty_module(full_name, make_type_void(NIL),l);
}

entity make_empty_function(const char* name, type r, language l)
{
  string full_name = concatenate(TOP_LEVEL_MODULE_NAME
				 MODULE_SEP_STRING, name, NULL);
  return make_empty_module(full_name, r,l);
}

entity make_empty_blockdata(const char* name,language l)
{
  string full_name = concatenate(TOP_LEVEL_MODULE_NAME MODULE_SEP_STRING
				 BLOCKDATA_PREFIX, name, NULL);
  return make_empty_module(full_name, make_type_void(NIL),l);
}


/* this function checks that e has an initial value code. if yes returns
it, otherwise aborts.  */

code EntityCode(entity e)
{
  value ve = entity_initial(e);
  pips_assert("EntityCode", value_tag(ve) == is_value_code);
  return(value_code(ve));
}

entity make_label(const char* module_name, const char* local_name)
{
    entity l = FindOrCreateEntity(module_name, local_name);
    if( type_undefined_p(entity_type(l)) ) {
        entity_type(l) =  MakeTypeStatement();
        entity_storage(l) =  make_storage_rom();
        entity_initial(l) = make_value_constant(make_constant_litteral());
    }
    return l;
}

/* Maximal value set for Fortran 77 */
static int init = 100000;

void reset_label_counter()
{
  init = 100000;
}
char* new_label_local_name(entity module)
{
  string local_name;
  const char *module_name ;
  const char * format;

  pips_assert( "module != 0", module != 0 ) ;

  if( module == entity_undefined ) {
    module_name = GENERATED_LABEL_MODULE_NAME ;
    format = "%d";
  }
  else {
    module_name = module_local_name(module) ;
    format = c_module_p(module)?LABEL_PREFIX "l%d":LABEL_PREFIX "%d";
  }
  --init;
  for(asprintf(&local_name, format, init);
      init >= 0 && !entity_undefined_p(FindEntity(module_name, local_name)) ;) {
    free(local_name);
    --init;
    asprintf(&local_name, format, init);
    /* loop */
  }
  if(init == 0) {
    pips_internal_error("no more available labels");
  }
  return local_name;
}

/* This function returns a new label */
entity make_new_label(entity module)
{
  /* FI: do labels have to be declared?*/
  /* FI: it's crazy; the name is usually derived from the entity
     by the caller and here the entity is retrieved from its name! */
  string local_name = new_label_local_name(module);
  const char * module_name = entity_undefined_p(module)?
      GENERATED_LABEL_MODULE_NAME:
      module_local_name(module);
  return make_label(module_name, local_name);

}

entity make_loop_label(int __attribute__ ((unused)) desired_number,
		entity module)
{
  entity e = make_new_label(module);
  return e;
}

static bool label_defined_in_statement = false;
static entity label_searched_in_statement = entity_undefined;

static bool check_statement_for_label(statement s)
{
  if(!label_defined_in_statement) {
    label_defined_in_statement = (statement_label(s)==label_searched_in_statement);
  }
  return !label_defined_in_statement;
}

bool label_defined_in_statement_p(entity l, statement s)
{
  label_defined_in_statement = false;
  label_searched_in_statement = l;

  gen_recurse(s, statement_domain, check_statement_for_label, gen_null);
  label_searched_in_statement = entity_undefined;

  return label_defined_in_statement;
}

bool label_defined_in_current_module_p(entity l)
{
  statement s = get_current_module_statement();
  bool defined_p = label_defined_in_statement_p(l, s);

  return defined_p;
}

bool label_string_defined_in_current_module_p(string ls)
{
  entity l = find_label_entity(get_current_module_name(), ls);
  pips_assert("entity defined",!entity_undefined_p(l));
  statement s = get_current_module_statement();
  bool defined_p = label_defined_in_statement_p(l, s);

  return defined_p;
}

bool label_string_defined_in_statement_p(string ls, statement s)
{
  entity l = find_label_entity(get_current_module_name(), ls);

  bool defined_p = label_defined_in_statement_p(l, s);

  return defined_p;
}

/* predicates and functions for entities
 */

/* BEGIN_EOLE */ /* - please do not remove this line */
/* Lines between BEGIN_EOLE and END_EOLE tags are automatically included
   in the EOLE project (JZ - 11/98) */

string safe_entity_name(entity e)
{
  string sn = string_undefined;

  if(entity_undefined_p(e))
    sn = "undefined object, entity assumed";
  else if(entity_domain_number(e)!= entity_domain)
    sn = "not an entity";
  else
    sn = entity_name(e);
  return sn;
}


/* entity_local_name modified so that it does not core when used in
 * vect_fprint, since someone thought that it was pertinent to remove the
 * special care of constants there. So I added something here, to deal
 * with the "null" entity which codes the constant. FC 28/11/94.
 * SG: should return a const pointer
 */
const char *
entity_local_name(entity e)
{
  const char* null_name = "null";
  pips_assert("entity is defined", !entity_undefined_p(e));
  pips_assert("constant term or entity",
	      e==NULL || entity_domain_number(e)==entity_domain);
  return e==NULL ? null_name : local_name(entity_name(e));
}


/* Used instead of the macro to pass as formal argument */
string entity_global_name(entity e)
{
  //string null_name = "null";
  pips_assert("entity is defined", !entity_undefined_p(e));
  return entity_name(e);
}

/* Since entity_local_name may contain PIPS special characters such as
   prefixes (label, common, struct, union, typedef, ...), this
   entity_user_name function is created to return the initial
   entity/variable name, as viewed by the user in his code.

   In addition, all possible seperators (file, module, block, member)
   are taken into account.

   Function strstr locates the occurence of the last special character
   which can appear just before the initial name, so the order of test
   is important.

   01/08/2003 Nga Nguyen -

   @return pointer to the the user name (not newly allocated!)
*/
const char * entity_user_name(entity e)
{
  string gn = entity_name(e);
  const char* un = global_name_to_user_name(gn);
  return un;
}

/* Functions used to manage the block scoping in conjunction with
   ContextStack and ycontext */

string empty_scope() { return strdup("");}

bool empty_scope_p(string s) {return strcmp(s, "")==0;}

/* same kind of testing required for union as well */
bool string_struct_scope_p(string s)
{
  /* Full testing would require a module_name, a block_scope and a struct name */
  /* Just lookup the struct identifier*/
  string ss = strchr(s, MEMBER_SEP_CHAR);
  return ss != NULL;
}

bool string_block_scope_p(string s)
{
  // A block scope string is empty or made of numbers each terminated by BLOCK_SEP_STRING
  char valid[12] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', BLOCK_SEP_CHAR, '\0'};
  bool is_block_scope = false;
  string cs = s;
  bool is_number = false;

  pips_debug(10, "Potential block scope string = \"%s\"\n", s);

  if(strspn(s, valid) == strlen(s)) {
    for(cs=s; *cs!='\0'; cs++) {
      if(is_number && isdigit(*cs))
	;
      else if(is_number && *cs==BLOCK_SEP_CHAR)
	is_number = false;
      else if(!is_number && isdigit(*cs))
	is_number = true;
      else if(!is_number && *cs==BLOCK_SEP_CHAR) {
	is_block_scope = false;
	break;
      }
    }
    is_block_scope = !is_number;
  }

  pips_debug(10, "String = \"%s\" is %sa block scope string\n", s, is_block_scope?"":"not ");

  return is_block_scope;
}

/* allocates a new string */
string entity_name_without_scope(entity e)
{
  string en = entity_name(e);
  const char* mn = entity_module_name(e);
  string ns = strrchr(en, BLOCK_SEP_CHAR);
  string enws = string_undefined;

  if(ns==NULL)
    enws = strdup(en);
  else
    enws = strdup(concatenate(mn, MODULE_SEP_STRING, ns+1, NULL));

  pips_debug(9, "entity name = \"%s\", without scope: \"%s\"\n",
	     en, enws);

  return enws;
}


/* allocates a new string */
string local_name_to_scope(const char* ln)
{
  string ns = strrchr(ln, BLOCK_SEP_CHAR);
  string s = string_undefined;

  if(ns==NULL)
    s = empty_scope();
  else
    s = strndup(ln, ns-ln+1);

  pips_debug(8, "local name = \"%s\",  scope: \"%s\"\n",
	     ln, s);

  return s;
}


/* Returns the module local user name
 */
const char* module_local_name(entity e)
{
  /* No difference between modules and other entities, except for prefixes */
  const char* name = local_name(entity_name(e));

  return (name
   + strspn(name, F95MODULE_PREFIX MAIN_PREFIX BLOCKDATA_PREFIX COMMON_PREFIX));
}

/* Returns a pointer towards the resource name. The resource name is
   the module local name: it may include the  */
const char* module_resource_name(entity e)
{
  const char* rn = entity_local_name(e);

  rn += strspn(rn, F95MODULE_PREFIX MAIN_PREFIX BLOCKDATA_PREFIX COMMON_PREFIX);

  return rn;
}

/* END_EOLE */

const char* label_local_name(entity e)
{
  const char* name = local_name(entity_name(e));
  return name+sizeof(LABEL_PREFIX) -1 ;
}

bool label_name_conflict_with_labels(const char* n, list ll)
{
  bool conflict_p = false;
  if(!empty_label_p(n)) {
    FOREACH(ENTITY, l, ll) {
      if(strcmp(label_local_name(l), n) == 0)
	conflict_p = true;
    }
  }
  return conflict_p;
}

/* Return a name valid for sorting variables in vectors and constraint
   systems.

   @return the name or "TCST" if the entity is null.
*/
const char* entity_name_or_TCST(entity e)
{
  if (e != NULL)
    return entity_name(e);
  else
    return "TCST";
}


/* See next function! */
/*
string
entity_relative_name(e)
entity e;
{
    entity m = get_current_module_entity();
    string s = string_undefined;

    pips_assert("entity_relative_name", !entity_undefined_p(m));

    s = (strcmp(module_local_name(m), entity_module_name(m)) == 0) ?
	entity_local_name(e) : entity_name(e) ;

    return s;
}
*/

const char* entity_and_common_name(entity e)
{
  entity m = get_current_module_entity();
  string name ;
  pips_assert("some current entity", !entity_undefined_p(m));

  name = concatenate(entity_local_name(ram_section(storage_ram(entity_storage(e)))),
		     MODULE_SEP_STRING,entity_name(e),NIL);

  return name +sizeof(COMMON_PREFIX) -1;
}

bool entity_empty_label_p(entity e)
{
  const char* lln = entity_local_name(e);
  bool empty_p = empty_label_p(lln);
  return empty_p;
}

bool entity_return_label_p(entity e)
{
  return return_label_p(entity_name(e));
}

bool entity_label_p(entity e)
{
  return type_statement_p(entity_type(e));
}

bool entity_module_p(entity e)
{
  if(typedef_entity_p(e))
    /* Functional typedef also have value code ... */
    return false;
  else {
    value v = entity_initial(e);
    return v!=value_undefined && value_code_p(v);
  }
}

bool entity_f95use_p(entity e)
{
  const char* name = entity_name(e);
  return strncmp(name,F95_USE_LOCAL_NAME,strlen(F95_USE_LOCAL_NAME)) == 0;
}

bool entity_main_module_p(entity e)
{
  return entity_module_p(e)
    && (strspn(entity_local_name(e), MAIN_PREFIX)==1
	|| same_string_p(entity_local_name(e), "main"));
}

bool entity_f95module_p(entity e) {
  return entity_module_p(e) &&
      strspn(entity_local_name(e), F95MODULE_PREFIX)==1;
}

bool entity_blockdata_p(entity e)
{
  return entity_module_p(e) &&
    strspn(entity_local_name(e), BLOCKDATA_PREFIX)==1;
}

bool entity_common_p(entity e)
{
  return entity_module_p(e) && /* ?????? */
    strspn(entity_local_name(e), COMMON_PREFIX)==1;
}

bool entity_function_p(entity e)
{
  type
    t_ent = entity_type(e),
    t = (type_functional_p(t_ent) ?
	 functional_result(type_functional(t_ent)) :
	 type_undefined);

  return(entity_module_p(e) &&
	 !type_undefined_p(t) &&
	 !type_void_p(t));
}

bool entity_subroutine_p(entity e)
{
  return entity_module_p(e) &&
    !entity_main_module_p(e) &&
    !entity_blockdata_p(e) && /* ??? */
    !entity_function_p(e);
}

bool entity_pointer_p(entity e)
{
    type t = ultimate_type(entity_type(e));
    basic b  = type_variable_p(t) ? variable_basic(type_variable(t)): basic_undefined;
    return !basic_undefined_p(b) && basic_pointer_p(b);
}

bool entity_array_p(entity e)
{
  if (entity_variable_p(e)) {
    return array_type_p(ultimate_type(entity_type(e)));
  }
  return false;
}

/* @return whether entity is a "register" variable
 *
 * See also volatile_variable_p()
 */
bool entity_register_p(entity e)
{
  if (entity_variable_p(e))
  {
    FOREACH(qualifier, q, variable_qualifiers(type_variable(entity_type(e))))
      if (qualifier_register_p(q))
        return true;
  }
  return false;
}

/* Assuming that v is of type variable, add a qualifier register */
void set_register_qualifier(entity v)
{
  if(!entity_register_p(v)) {
    //type uvt = ultimate_type(entity_type(v))
    type vt = entity_type(v);
    if(type_variable_p(vt)) {
      list *ql = &variable_qualifiers(type_variable(vt));
      qualifier q = make_qualifier_register();
      *ql =	gen_nconc(*ql, CONS(QUALIFIER, q , NIL));
    }
    else
      pips_internal_error("Improper argument\n");
  }
}

bool array_entity_p(entity e)
{
    return entity_array_p(e);
}

bool entity_variable_length_array_p(entity e) {
  bool return_val = false;
  if (entity_variable_p(e)) {
    return_val=variable_length_array_type_p(ultimate_type(entity_type(e)));
  }
  return return_val;
}


bool assumed_size_array_p(entity e)
{
  /* return true if e has an assumed-size array declarator
     (the upper bound of the last dimension is equal to * : REAL A(*) )*/
  if (entity_variable_p(e))
    {
      variable v = type_variable(entity_type(e));
      list l_dims = variable_dimensions(v);
      if (l_dims != NIL)
	{
	  int length = gen_length(l_dims);
	  dimension last_dim =  find_ith_dimension(l_dims,length);
	  if (unbounded_dimension_p(last_dim))
	    return true;
	}
    }
  return false;
}

bool pointer_type_array_p(entity e)
{
  /* return true if e has a pointer-type array declarator
     (the upper bound of the last dimension is  equal to 1: REAL A(1) )*/
  if (entity_variable_p(e))
    {
      variable v = type_variable(entity_type(e));
      list l_dims = variable_dimensions(v);
      if (l_dims != NIL)
	{
	  int length = gen_length(l_dims);
	  dimension last_dim =  find_ith_dimension(l_dims,length);
	  expression exp = dimension_upper(last_dim);
	  if (expression_equal_integer_p(exp,1))
	    return true;
	}
    }
  return false;
}

bool unnormalized_array_p(entity e)
{
  /* return true if e is an assumed-size array or a pointer-type array*/
  if (assumed_size_array_p(e) || pointer_type_array_p(e))
    return true;
  return false;
}



/* e is the field of a structure */
bool entity_field_p(entity e)
{
  const char* eln = entity_local_name(e);
  bool field_p = false;

  if(*eln!='\'' && *eln!='"') {
    const char* pos = strrchr(eln, MEMBER_SEP_CHAR);

    field_p = pos!=NULL;
  }

  return field_p;
}

/* f is a field of a structure: what is its structure?
 *
 * To get the structure name, we have to drop the field part of f's
 * name and to insert a struct prefix before the struct name. Maybe,
 * it would have been better to keep the struct prefix in the field
 * name.
 */
static
entity entity_field_to_entity(entity f, char prefix)
{
  entity s = entity_undefined;
  const string sn = strdup(entity_name(f)); /* structure name */
  string pos = strrchr(sn, MEMBER_SEP_CHAR);
  string usn = string_undefined;
  int usnl = 0;

  pips_assert("The entity is a field", pos!=NULL);

  *pos = '\0'; /* get rid of the field name */
  usn = strdup(global_name_to_user_name(sn));
  usnl = strlen(usn);
  *(pos-usnl) = prefix;
  /* can be done in place because the field name is at least on
     character long and because we also gain the field marker */
  (void) strncpy(pos-usnl+1, usn, usnl+1);
  free(usn);

  pips_debug(8, "struct entity name is \"\%s\"\n", sn);
  s = gen_find_tabulated(sn, entity_domain);
  free(sn);

  return s;
}

entity entity_field_to_entity_struct(entity f)
{
  entity s = entity_field_to_entity(f, STRUCT_PREFIX_CHAR);

  /* To be able to breakpoint effectively */
  if(entity_undefined_p(s))
    pips_assert("entity s is defined", !entity_undefined_p(s));
  return s;
}

entity entity_field_to_entity_union(entity f)
{
  entity u = entity_field_to_entity(f, UNION_PREFIX_CHAR);

  /* To be able to breakpoint effectively */
  if(entity_undefined_p(u))
    pips_assert("entity s is defined", !entity_undefined_p(u));
  return u;
}

entity entity_field_to_entity_struct_or_union(entity f)
{
  entity su = entity_field_to_entity(f, UNION_PREFIX_CHAR);

  if(entity_undefined_p(su))
    su = entity_field_to_entity(f, STRUCT_PREFIX_CHAR);

  /* To be able to breakpoint effectively */
  if(entity_undefined_p(su))
    pips_assert("entity s is defined", !entity_undefined_p(su));

  return su;
}

/* f is a field of a structure or of an union: what is its rank? */
int entity_field_rank(entity f)
{
  int rank = -1;
  entity su = entity_field_to_entity_struct_or_union(f);
  type st = entity_type(su);
  list fl = list_undefined;

  if(type_struct_p(st))
    fl = type_struct(st);
  else if(type_union_p(st))
    fl = type_union(st);
  else
    pips_internal_error("Unexpected type tag %d", type_tag(st));

  pips_assert("st is a struct or union type",
	      type_struct_p(st) || type_union_p(st));

  /* FI: positions are counted from 1 on; do we want to subtract 1? */
  rank = gen_position((void *) f, fl);

  if(rank==0) {
    pips_internal_error("Field \"\%s\" is not part of its %s \"\%s\"",
			entity_name(f), type_struct_p(st)?"structure":"union" , entity_name(su));
  }

  return rank;
}

bool entity_enum_p(entity e)
{
  /* Base the predicate on the entity name as for struct and union.*/
  //return type_enum_p(entity_type(e));
  const char* ln = entity_local_name(e);
  string ns = strrchr(ln, BLOCK_SEP_CHAR);
  bool struct_p = (ns==NULL && *ln==ENUM_PREFIX_CHAR)
    || (ns!=NULL && *(ns+1)==ENUM_PREFIX_CHAR)
    || (strstr(entity_name(e),ENUM_PREFIX DUMMY_ENUM_PREFIX)!=NULL);
  return struct_p;
}

bool entity_enum_member_p(entity e)
{
  value ev = entity_initial(e);
  /* SG: not all entities seem to have this field defined if not
   * defined, assume it's not an enum, although i am unsure of the
   * validity of this:
   *
   * pips_assert("Value of e is defined", !value_undefined_p(ev));
   */
  return !value_undefined_p(ev) && value_symbolic_p(ev);
}

bool entity_enum_variable_p(entity e) {
    type t = ultimate_type(entity_type(e));
    basic b  = type_variable_p(t) ? variable_basic(type_variable(t)): basic_undefined;
    return !basic_undefined_p(b) && basic_derived_p(b) && entity_enum_p(basic_derived(b));
}

bool entity_struct_p(entity e)
{
  const char* ln = entity_local_name(e);
  string ns = strrchr(ln, BLOCK_SEP_CHAR);
  bool struct_p = (ns==NULL && *ln==STRUCT_PREFIX_CHAR)
    || (ns!=NULL && *(ns+1)==STRUCT_PREFIX_CHAR)
    || (strstr(entity_name(e),STRUCT_PREFIX DUMMY_STRUCT_PREFIX)!=NULL);
  return struct_p;
}

bool same_struct_entity_p(const entity e0, const entity e1)
{
    entity s0 = entity_field_to_entity_struct(e0),
           s1 = entity_field_to_entity_struct(e1);
    return same_entity_p(s0,s1);
}

bool entity_union_p(entity e)
{
  const char* ln = entity_local_name(e);
  string ns = strrchr(ln, BLOCK_SEP_CHAR);
  bool union_p = (ns==NULL && *ln==UNION_PREFIX_CHAR)
    || (ns!=NULL && *(ns+1)==UNION_PREFIX_CHAR)
    || (strstr(entity_name(e),UNION_PREFIX DUMMY_UNION_PREFIX)!=NULL);
  return union_p;
}

bool derived_entity_p(entity e)
{
  return entity_struct_p(e) || entity_union_p(e) || entity_enum_p(e);
}

/* This test shows that "e" has been declared in "module".
 *
 * Well, "e" may not be declared in "module". For instance, "e" may be
 * a value of a variable declared in "module".
 *
 * This does not show in Fortran that e is a variable with effects local
 * to the module because e can be allocated in a common. Variables with
 * local effects are allocated either in the static or the dynamic or the
 * stack area.
 *
 * Variables with effects lost on return are allocated either in the
 * dynamic or stack areas. Effects on static variables may or not escape.
 *
 * Of course, this predicate returns false for some variables declared
 * in "module", extern variables for instance.
 */
bool local_entity_of_module_p(entity e, entity module)
{
  bool
    result = same_string_p(entity_module_name(e),
			   module_local_name(module));

  debug(6, "local_entity_of_module_p",
	"%s %s %s\n",
	entity_name(e), result ? "in" : "not in", entity_name(module));

  return(result);
}

bool entity_in_common_p(entity e)
{
  storage s = entity_storage(e);

  return(storage_ram_p(s) &&
	 !entity_special_area_p(ram_section(storage_ram(s))));
}

/* See comments about module_name(). Its result is transient and must
   be strduped. */
const char* entity_module_name(entity e)
{
  return module_name(entity_name(e));
}


code entity_code(entity e)
{
  value ve = entity_initial(e);
  pips_assert("entity_code",value_code_p(ve));
  return(value_code(ve));
}

entity entity_empty_label(void)
{
  /* FI: it is difficult to memoize entity_empty_label because its value is changed
   * when the symbol table is written and re-read from disk; Remi's memoizing
   * scheme was fine as long as the entity table lasted as long as one run of
   * pips, i.e. is not adequate for wpips
   */
  /* static entity empty = entity_undefined; */
  entity empty;

  empty = gen_find_tabulated(concatenate(TOP_LEVEL_MODULE_NAME,
					 MODULE_SEP_STRING,
					 EMPTY_LABEL_NAME,
					 NULL), entity_domain);
  pips_assert("entity_empty_label", empty != entity_undefined );

  return empty;
}

/* Check if the scope of entity e is global.

   People are likely to look for global_entity_p(), entity_global_p()
   or, slightly different, global_variable_p(), variable_global_p(),
   or still slightly different, global_function_p(),...
 */
bool top_level_entity_p(entity e)
{
  bool top = (strcmp(TOP_LEVEL_MODULE_NAME, entity_module_name(e)) == 0);

  return top;
}

/* Several implicit entities are declared to define the implicit
   effects of IO statements. */
bool io_entity_p(entity e)
{
  set_internal_static_entities();
  return (same_entity_p(e, luns_ent) || same_entity_p(e, io_ptr_ent)
	  || same_entity_p(e, io_eof_ent) || same_entity_p(e, io_error_luns_ent));
}

bool io_luns_entity_p(entity e)
{
  set_internal_static_entities();
  return (same_entity_p(e, luns_ent));
}

bool rand_effects_entity_p(entity e)
{
  set_internal_static_entities();
  return (same_entity_p(e, rand_gen_ent));
}

bool malloc_effect_entity_p(entity e)
{
  set_internal_static_entities();
  return (same_entity_p(e, malloc_effect_ent));

}

bool memmove_effect_entity_p(entity e) {
  set_internal_static_entities();
  return (same_entity_p(e, memmove_effect_ent));
}

bool time_effect_entity_p(entity e) {
  set_internal_static_entities();
  return (same_entity_p(e, time_effect_ent));
}

/**
   checks if an entity is an IO_EFFECTS_PACKAGE_NAME, a
   MALLOC_EFFECTS_NAME or a RAND_EFFECTS_PACKAGE_NAME entity. These
   entities are used to model some internal effects of standard libraries
   and they do not conflict with other entities.
 */
bool effects_package_entity_p(entity e)
{
#ifndef NDEBUG
    bool result = rand_effects_entity_p(e) 
        || malloc_effect_entity_p(e)
        || memmove_effect_entity_p(e)
        || time_effect_entity_p(e)
        || io_entity_p(e);
    pips_assert("entity kind is consistent", result == ((entity_kind(e) & EFFECTS_PACKAGE) == EFFECTS_PACKAGE));
#endif
  return entity_kind(e) & EFFECTS_PACKAGE;
}




entity get_stdin_entity()
{
  set_std_static_entities();
  return stdin_ent;
}

bool stdin_entity_p(entity e)
{
  return same_entity_p(e, get_stdin_entity());
}


entity get_stdout_entity()
{
  set_std_static_entities();
  return stdout_ent;
}

bool stdout_entity_p(entity e)
{
  return same_entity_p(e, get_stdout_entity());
}

entity get_stderr_entity()
{
  set_std_static_entities();
  return stderr_ent;
}

bool stderr_entity_p(entity e)
{
  return same_entity_p(e, get_stderr_entity());
}


bool std_file_entity_p(entity e)
{
  set_std_static_entities();
  return(same_entity_p(e, stdin_ent)
	 || same_entity_p(e, stdout_ent)
	 || same_entity_p(e, stderr_ent));
}




bool intrinsic_entity_p(entity e)
{
  return (!value_undefined_p(entity_initial(e)) && value_intrinsic_p(entity_initial(e)));
}

bool symbolic_entity_p(entity e)
{
  return (!value_undefined_p(entity_initial(e)) && value_symbolic_p(entity_initial(e)));
}

bool intrinsic_name_p(const char *local_name) {
    entity e = FindEntity(TOP_LEVEL_MODULE_NAME, local_name);
    return !entity_undefined_p(e) && intrinsic_entity_p(e);
}

/* FI: I do not understand this function name (see next one!). It seems to me
 * that any common or user function or user subroutine would
 * be returned.
 * FI: assert condition made stronger (18 December 1998)
 */
entity entity_intrinsic(const char* name)
{
  entity e = (entity) gen_find_tabulated(concatenate(TOP_LEVEL_MODULE_NAME,
						     MODULE_SEP_STRING,
						     name,
						     NULL),
					 entity_domain);

  pips_assert("entity_intrinsic", e != entity_undefined
	      && intrinsic_entity_p(e));
  return(e);
}



/* this function does not create an intrinsic function because they must
   all be created beforehand by the bootstrap phase (see
   bootstrap/bootstrap.c). */

entity CreateIntrinsic(string name)
{
  entity e = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME, name);
  pips_assert("entity is defined", e!=entity_undefined );
  pips_assert("entity is intrinsic", intrinsic_entity_p(e));
  return(e);
}

/* predicates on entities */

bool same_entity_p(entity e1, entity e2)
{
  return(e1 == e2);
}

/*  Comparison function for qsort.
 */
int compare_entities(const entity *pe1, const entity *pe2)
{
  int
    null_1 = (*pe1==(entity)NULL),
    null_2 = (*pe2==(entity)NULL);

  if (null_1 || null_2)
    // FI: I reverse the test to place the constant term at the end of
    //the vector so as to regenerate expressions with trailing
    //constant terms; for instance, I get J+1 instead of 1+J.
    // Of course, this impacts PIPS code generation
    //return(null_2-null_1);
    return(null_1-null_2);
  else {
    /* FI: Which sorting do you want? */

    //string s1 = entity_name_without_scope(*pe1);
    //string s2 = entity_name_without_scope(*pe2);
    //int c = strcmp(s1, s2);
    //
    //free(s1);
    //free(s2);
    //
    //return c;
    return strcmp(entity_name(*pe1), entity_name(*pe2));
  }
}

/* sorted in place.
 */
void sort_list_of_entities(list l)
{
  gen_sort_list(l, (gen_cmp_func_t)compare_entities);
}

/*   true if var1 <= var2
 */
bool lexicographic_order_p(entity var1, entity var2)
{
  /*   TCST is before anything else
   */
  if ((Variable) var1==TCST) return(true);
  if ((Variable) var2==TCST) return(false);

  /* else there are two entities
   */

  return(strcmp(entity_local_name(var1), entity_local_name(var2))<=0);
}

/* return the basic associated to entity e if it's a function/variable/constant
 * basic_undefined otherwise */
basic entity_basic(entity e)
{
  if (e != entity_undefined) {
    type t = entity_type(e);

    if (type_functional_p(t))
      t = functional_result(type_functional(t));
    if (type_variable_p(t))
      return (variable_basic(type_variable(t)));
  }
  return (basic_undefined);
}
/* return the qualifiers associated to entity e if it's a variable
 * NIL otherwise */
list entity_qualifiers(entity e)
{
  if (e != entity_undefined) {
    type t = entity_type(e);
    if (type_variable_p(t))
      return (variable_qualifiers(type_variable(t)));
  }
  return NIL;
}

/* return true if the basic associated with entity e matchs the passed tag */
bool entity_basic_p(entity e,enum basic_utype basictag)
{
  return basic_tag(entity_basic(e)) == basictag;
}

/* Checks that el only contains entity*/
bool entity_list_p(list el)
{
  bool pure = true;

  FOREACH(ENTITY, e, el)
      {
	static entity le = entity_undefined;
	pips_debug(10, "Entity e in list is \"%s\"\n", safe_entity_name(e));
	if(entity_domain_number(e)!=entity_domain) {
	  pips_debug(8, "Last entity le in list is \"%s\"\n", safe_entity_name(le));
	  pure = false;
	  break;
	}
	le = e;
      }
  return pure;
}

/* this function maps a local name, for instance P, to the corresponding
 * TOP-LEVEL entity, whose name is TOP-LEVEL:P. n is the local name.
 */
#define PREFIXES_SIZE 5
static string prefixes[] = {
    "",
    MAIN_PREFIX,
    BLOCKDATA_PREFIX,
    COMMON_PREFIX,
    F95MODULE_PREFIX,
};


/**
 * @brief This function try to find a top-level entity from a local name
 *
 * @description Because of static C function, the entity returned is not always
 * a top-level entity.
 *
 * @return the entity if found, else entity_undefined
 */
entity local_name_to_top_level_entity(const char *n)
{
  entity module = entity_undefined;

  /* Extension with C: the scope of a module can be its compilation unit if this is
     a static module, not only TOP-LEVEL. */

  if (static_module_name_p(n)) {
    string cun = strdup(n);
    string sep = strchr(cun, FILE_SEP);
    *(sep+1) = '\0';
    module = FindEntity(cun,n);
    free(cun);
  }
  else
    {
      for(int i=0; i<PREFIXES_SIZE && entity_undefined_p(module); i++)
        module = gen_find_tabulated(concatenate
				    (TOP_LEVEL_MODULE_NAME, MODULE_SEP_STRING, prefixes[i], n, NULL),
				    entity_domain);
    }

  return module;
}

/**
 * @brief This is an alias for local_name_to_top_level_entity
 * @return the entity if found, else entity_undefined
 */
entity module_name_to_entity(const char* mn) {
  return local_name_to_top_level_entity(mn);
}
/* similar to module_name_to_entity
 * but generates a warning and a stub if the entity is not found
 */
entity module_name_to_runtime_entity(const char* name)
{
    entity e = module_name_to_entity(name); 
    if ( entity_undefined_p( e ) )
    {
        pips_user_warning("entity %s not defined, pips is likely to crash soon\n"
                "Please feed pips with its definition and source\n",name);
        e = make_empty_subroutine(name,copy_language(module_language(get_current_module_entity())));
    }

    return e;
}


/**
 * @brief Retrieve an entity from its package/module name and its local name
 * @return the entity if found, else entity_undefined
 */
entity FindEntity(const char* package, const char* name ) {
  return gen_find_tabulated( concatenate( package,
                                          MODULE_SEP_STRING,
                                          name,
                                          NULL ),
          entity_domain );
}

entity FindEntityFromUserName( const char* package, const char* name ) {
  entity e = FindEntity(package, name);
  if ( entity_undefined_p(e) ) {
    e = gen_find_tabulated( concatenate( package,
					 MODULE_SEP_STRING "0" BLOCK_SEP_STRING,
					 name,
					 NULL ),
            entity_domain );
  }
  if ( entity_undefined_p(e) ) {
    e = gen_find_tabulated( concatenate( package,
					 MODULE_SEP_STRING ,
					 name,
					 NULL ),
            entity_domain );
  }
  return e;
}


/* BEGIN_EOLE */ /* - please do not remove this line */
/* Lines between BEGIN_EOLE and END_EOLE tags are automatically included
   in the EOLE project (JZ - 11/98) */

/*
 * Cette fonction est appelee chaque fois qu'on rencontre un nom dans le texte
 * du pgm Fortran.  Ce nom est celui d'une entite; si celle-ci existe deja on
 * renvoie son indice dans la table des entites, sinon on cree une entite vide
 * qu'on remplit partiellement.
 *
 * full_name est dupliqué.
 *
 * Modifications:
 *  - partial link at parse time for global entities (Remi Triolet?)
 *  - partial link limited to non common variables (Francois Irigoin,
 *    25 October 1990); see below;
 *  - no partial link: no information is available about "name"'s type; it
 *    can be a local variable as well as a global one; if D is a FUNCTION and
 *    D is parsed and put in the database, any future D would be interpreted
 *    as a FUNCTION and an assignment like D = 0. would generate a parser
 *    error message (Francois Irigoin, 6 April 1991)
 *  - partial link limited to intrinsic: it's necessary because there is no
 *    real link; local variables having the same name as an intrinsic will
 *    cause trouble; I did not find a nice way to fix the problem later,
 *    as it should be, in update_called_modules(), MakeAtom() or MakeCallInst()
 *    it would be necessary to rewrite the link phase; (Francois Irigoin,
 *    11 April 1991)
 *  - no partial link at all: this is incompatible with the Perfect Club
 *    benchmarks; variables DIMS conflicts with intrinsics DIMS;
 *    (Francois Irigoin, ?? ???? 1991)
 */

entity CreateEntity(const char *package_name, const char * local_name)
{
    char * name;
    asprintf(&name,"%s"MODULE_SEP_STRING"%s",package_name, local_name);
    entity e = make_entity(name, type_undefined, storage_undefined, value_undefined);
    return e;
}



/* Problem: A functional global entity may be referenced without
   parenthesis or CALL keyword in a function or subroutine call.
   See SafeFindOrCreateEntity().
*/
entity FindOrCreateEntity(const char* package /* package name */,
			  const char* local_name /* entity name */)
{
  entity e;
  if(entity_undefined_p(e=FindEntity(package, local_name))) {
      e = CreateEntity(package,local_name);
  } 
  return e;
}


/* Return a top-level entity

   @param name of the entity to find/construct

   @return the entity
*/
entity FindOrCreateTopLevelEntity(const char* name)
{
  return FindOrCreateEntity(TOP_LEVEL_MODULE_NAME, name);
}


/* FIND_MODULE returns entity. Argument is module_name */
/* This function should be replaced by local_name_to_top_level_entity() */
/*entity FindEntity(_module(name)
string name;
{
    string full_name = concatenate(TOP_LEVEL_MODULE_NAME,
				   MODULE_SEP_STRING, name, NULL);
    entity e = gen_find_tabulated(full_name, entity_domain);

    return(e);
}
*/


/* END_EOLE */

/* returns a range expression containing e's i-th bounds */
expression entity_ith_bounds(entity e, int i)
{
  dimension d = entity_ith_dimension(e, i);
  syntax s = make_syntax(is_syntax_range,
			 make_range(copy_expression(dimension_lower(d)),
				    copy_expression(dimension_upper(d)),
				    int_to_expression(1)));
  return(make_expression(s, normalized_undefined));
}


/*true is a statement s is an io intrinsic*/
/*bool statement_contains_io_intrinsic_call_p(statement s)
{
 IoElementDescriptor *pid = IoElementDescriptorTable;
      bool found = false;

      while ((pid->name != NULL) && (!found)) {
	if (strcmp(pid->name, s) == 0)
	  {
	    found = true;
	    return true;
	  }
      }
      return false;
}*/

/* true if e is an io instrinsic
 */
bool io_intrinsic_p(entity e)
{
  if (set_undefined_p(io_functions_set)) {
    io_functions_set = set_make(set_pointer);
    set_add_elements(io_functions_set, io_functions_set, entity_intrinsic(SCANF_FUNCTION_NAME),
		     entity_intrinsic(PRINTF_FUNCTION_NAME),
		     entity_intrinsic(SCANF_FUNCTION_NAME),
		     entity_intrinsic(ISOC99_SCANF_FUNCTION_NAME),
		     entity_intrinsic(FPRINTF_FUNCTION_NAME),
		     entity_intrinsic(ISOC99_SCANF_USER_FUNCTION_NAME),
		     entity_intrinsic(PUTS_FUNCTION_NAME),
		     entity_intrinsic(GETS_FUNCTION_NAME),
		     entity_intrinsic(FOPEN_FUNCTION_NAME),
		     entity_intrinsic(FCLOSE_FUNCTION_NAME),
		     entity_intrinsic(SNPRINTF_FUNCTION_NAME),
		     entity_intrinsic(SSCANF_FUNCTION_NAME),
		     entity_intrinsic(ISOC99_SSCANF_FUNCTION_NAME),
		     entity_intrinsic(ISOC99_SSCANF_USER_FUNCTION_NAME),
		     entity_intrinsic(VFPRINTF_FUNCTION_NAME),
		     entity_intrinsic(VFSCANF_FUNCTION_NAME),

		     /*Fortran*/
		     entity_intrinsic(WRITE_FUNCTION_NAME),
		     //entity_intrinsic(PRINT_FUNCTION_NAME),
		     entity_intrinsic(REWIND_FUNCTION_NAME),
		     entity_intrinsic(OPEN_FUNCTION_NAME),
		     entity_intrinsic(CLOSE_FUNCTION_NAME),
		     entity_intrinsic(INQUIRE_FUNCTION_NAME),
		     entity_intrinsic(BACKSPACE_FUNCTION_NAME),
		     entity_intrinsic(READ_FUNCTION_NAME),
		     entity_intrinsic(BUFFERIN_FUNCTION_NAME),
		     entity_intrinsic(ENDFILE_FUNCTION_NAME),
		     entity_intrinsic(FORMAT_FUNCTION_NAME),
		     NULL);
  }
  if(set_belong_p(io_functions_set, e))
    return true;
  else
    return false;
}

/* true if e is an arithmetic instrinsic
 *
 * Used to determine if a logical argument must be promoted to integer
 *
 * FI: the arithmetic operator set is not fully defined. To be completed.
 */
bool arithmetic_intrinsic_p(entity e)
{
  if (set_undefined_p(arithmetic_functions_set)) {
    arithmetic_functions_set = set_make(set_pointer);
    set_add_elements(arithmetic_functions_set, arithmetic_functions_set,
		     entity_intrinsic(PLUS_OPERATOR_NAME),
		     entity_intrinsic(PLUS_C_OPERATOR_NAME),
		     entity_intrinsic(MINUS_OPERATOR_NAME),
		     entity_intrinsic(MINUS_C_OPERATOR_NAME),
		     entity_intrinsic(UNARY_PLUS_OPERATOR_NAME),
		     entity_intrinsic(UNARY_MINUS_OPERATOR_NAME),
		     entity_intrinsic(MULTIPLY_OPERATOR_NAME),
		     entity_intrinsic(DIVIDE_OPERATOR_NAME),
		     NULL);
  }
  if(set_belong_p(arithmetic_functions_set, e))
    return true;
  else
    return false;
}

/* true if continue. See also macro ENTITY_CONTINUE_P
 */

entity get_continue_entity()
{
  set_internal_static_entities();
  return continue_ent;
}
bool entity_continue_p(entity f)
{
  return same_entity_p(f, get_continue_entity());
}


/**************************************************** CHECK COMMON INCLUSION */

/* returns the list of entity to appear in the common declaration.
 */
list /* of entity */ common_members_of_module(entity common,
					      entity module,
					      bool only_primary /* not the equivalenced... */)
{
  list result = NIL;
  int cumulated_offset = 0;
  pips_assert("entity is a common", entity_area_p(common));

  list ld =  area_layout(type_area(entity_type(common)));
  entity v = entity_undefined;

  for(; !ENDP(ld);ld = CDR(ld))
    {
      v = ENTITY(CAR(ld));
      storage s = entity_storage(v);
      ram r;
      pips_assert("storage ram", storage_ram_p(s));
      r = storage_ram(s);
      if (ram_function(r)==module)
	{
	  int offset = ram_offset(r);
	  int size = 0;

	  if(heap_area_p(ram_section(r))) {
	    size = 0;
	  }
	  else if(stack_area_p(ram_section(r))) {
	    size = 0;
	  }
	  else {
	    if(!SizeOfArray(v, &size)) {
	      pips_internal_error("Varying size array \"%s\"", entity_name(v));
	    }
	  }

	  if (cumulated_offset==offset || !only_primary)
	    result = CONS(ENTITY, v, result);
	  else
	    break; /* drop equivalenced that come hereafter... */

	  cumulated_offset+=size;
	}
    }

  return gen_nreverse(result);
}

/* returns true if l contains an entity with same type, local name and offset.
 */
static bool comparable_entity_in_list_p(entity common, entity v, list l)
{
  entity ref = entity_undefined;
  bool ok, sn, so = false, st = false;

  /* first find an entity with the same NAME.
   */
  const char* nv = entity_local_name(v);
  MAP(ENTITY, e,
      {
	if (same_string_p(entity_local_name(e),nv))
	  {
	    ref = e;
	    break;
	  }
      },
      l);

  ok = sn = !entity_undefined_p(ref);

  pips_assert("v storage ram", storage_ram_p(entity_storage(v)));
  if (ok) pips_assert("ref storage ram", storage_ram_p(entity_storage(ref)));

  /* same OFFSET?
   */
  if (ok) ok = so = (ram_offset(storage_ram(entity_storage(v))) ==
		     ram_offset(storage_ram(entity_storage(ref))));

  /* same TYPE?
   */
  /* SG we cannot rely on same_type_p or type_equal_p because we want a syntactic comparison,
   * there used to be a hack in expression_equal_p to handle this particular case, I prefer to have the hack right here
   */
  if( ok ) {
      if(entity_undefined_p(ref)) ok = st =false;
      else {
          type tv = entity_type(v),
               tref = entity_type(ref);
          ok = st = same_type_name_p(tv,tref);
      }
  }

  pips_debug(4, "%s ~ %s? %d: n=%d,o=%d,t=%d\n", entity_name(v),
	     entity_undefined_p(ref)? "<undef>": entity_name(ref),
	     ok, sn, so, st);

  /* temporary for CA
   */
  if (!ok) {
    pips_debug(1, "common /%s/: %s != %s (n=%d,o=%d,t=%d)\n",
	       entity_name(common), entity_name(v),
	       entity_undefined_p(ref)? "<undef>": entity_name(ref),
	       sn, so, st);
  }

  return ok;
}

/* check whether a common declaration can be simply included, that is
 * it is declared with the same names, orders and types in all instances.
 */
bool check_common_inclusion(entity common)
{
  bool ok = true;
  list /* of entity */ lv, lref;
  entity ref;
  pips_assert("entity is a common", entity_area_p(common));
  lv = area_layout(type_area(entity_type(common)));

  if (!lv) return true; /* empty common! */

  /* take the first function as the reference for the check. */
  ref = ram_function(storage_ram(entity_storage(ENTITY(CAR(lv)))));
  lref = common_members_of_module(common, ref, false);

  /* SAME name, type, offset */
  while (lv && ok)
    {
      entity v = ENTITY(CAR(lv));
      if (ram_function(storage_ram(entity_storage(v)))!=ref)
	ok = comparable_entity_in_list_p(common, v, lref);
      POP(lv);
    }

  gen_free_list(lref);
  return ok;
}

/* This function creates a common for a given name in a given module.
   This is an entity with the following fields :
   Example:  SUBROUTINE SUB1
             COMMON /FOO/ W1,V1

   name = top_level:~name (TOP-LEVEL:~FOO)
   type = area
          with size = 8 [2*8], layout = NIL [SUB1:W,SUB1:V]
   storage = ram
          with function = module (TOP-LEVEL:SUB1) (first occurence ? SUB2,SUB3,..)
               section = TOP-LEVEL:~FOO  (recursive ???)
               offset = undefined
               shared = NIL
  initial = unknown

  The area size and area layout must be updated each time when
  a common variable is added to this common */

entity make_new_common(string name, entity mod)
{
  string common_global_name = strdup(concatenate(TOP_LEVEL_MODULE_NAME,
						 MODULE_SEP_STRING
						 COMMON_PREFIX,name,NULL));
  type common_type = make_type(is_type_area, make_area(8, NIL));
  entity StaticArea =
    FindOrCreateEntity(TOP_LEVEL_MODULE_NAME, STATIC_AREA_LOCAL_NAME);
  storage common_storage = make_storage(is_storage_ram,
					(make_ram(mod,StaticArea, 0, NIL)));
  value common_value =
    make_value_code(make_code(NIL,
			      string_undefined,
			      make_sequence(NIL),
			      NIL,
			      make_language_fortran()));

  return make_entity(common_global_name,
		     common_type,
		     common_storage,
		     common_value);
}

/* This function creates a common variable in a given common in a given module.
   This is an entity with the following fields :
   name = module_name:name (SUB1:W1)
   type = variable
          with basic = int, dimension = NIL
   storage = ram
          with function = module (TOP-LEVEL:SUB1)
               section = common (TOP-LEVEL:~FOO)
               offset = 0
               shared =
  initial = unknown

  The common must be updated with new area size and area layout */

entity make_new_integer_scalar_common_variable(string name, entity mod, entity com)
{
  string var_global_name = strdup(concatenate(module_local_name(mod),MODULE_SEP_STRING,
                name,NULL));
  type var_type = make_type(is_type_variable, make_variable(make_basic_int(8), NIL,NIL));
  storage var_storage = make_storage(is_storage_ram,
             (make_ram(mod,com,0,NIL)));
  value var_value = make_value_unknown();
  entity e = make_entity(var_global_name,var_type,var_storage,var_value);
  //area_layout(type_area(entity_type(com))) = CONS(ENTITY,e,NIL);
  return e;
}


#define declaration_formal_p(E) storage_formal_p(entity_storage(E))
#define entity_to_offset(E) formal_offset(storage_formal(entity_storage(E)))

/* This function gives back the ith formal parameter, which is found in the
 * declarations of a call or a subroutine.
 */
entity find_ith_formal_parameter(entity the_fnct, int rank)
{
  list ldecl = code_declarations(value_code(entity_initial(the_fnct)));
  entity current = entity_undefined;

  while (ldecl != NULL)
    {
      current = ENTITY(CAR(ldecl));
      ldecl = CDR(ldecl);
      if (declaration_formal_p(current) && (entity_to_offset(current)==rank))
	return current;
    }

  pips_internal_error("cannot find the %d dummy argument of %s",
		      rank, entity_name(the_fnct));

  return entity_undefined;
}

/* returns whether there is a main in the database
 */

bool some_main_entity_p(void)
{
  gen_array_t modules = db_get_module_list();
  bool some_main = false;
  GEN_ARRAY_MAP(name,
		if (entity_main_module_p(local_name_to_top_level_entity(name)))
		  some_main = true,
		modules);
  gen_array_full_free(modules);
  return some_main;
}

/* @return the list of entities in module the name of which is given
 * warning: the entity is created if it does not exist!
 * @param module the name of the module for the entities
 * @param names a string of comma-separated of entity names
 */
list /* of entity */ string_to_entity_list(string module, string names)
{
  list le = NIL;
  string s, next_comma = (char*) 1;
  for (s = names; s && *s && next_comma;)
    {
      next_comma = strchr(s, ',');
      if (next_comma) *next_comma = '\0';
      le = CONS(ENTITY, FindOrCreateEntity(module, s), le);
      s += strlen(s)+1;
      if (next_comma) *next_comma = ',';
    }
  return le;
}

bool typedef_entity_p(entity e)
{
  /* Its name must contain the TYPEDEF_PREFIX just after the
     MODULE_SEP_STRING and the scope information */
  string en = entity_name(e);
  string ms = strchr(en, BLOCK_SEP_CHAR);
  bool is_typedef = false;

  /* If there is no scope information, use the module separator */
  if(ms==NULL)
    ms = strchr(en, MODULE_SEP);


  if(ms!=NULL)
    is_typedef = (*(ms+1)==TYPEDEF_PREFIX_CHAR);

  return is_typedef;
}

bool member_entity_p(entity e)
{
  /* Its name must contain the MEMBER_PREFIX after the MODULE_SEP_STRING */
  string en = entity_name(e);
  string ms = strchr(en, MODULE_SEP);
  bool is_member = false;

  if(ms!=NULL)
    is_member = (strchr(ms, MEMBER_SEP_CHAR)!=NULL);

  return is_member;
}

/* is p a formal parameter? */
bool entity_formal_p(entity p)
{
    return formal_parameter_p(p);
}

/* is p a dummy parameter? */
bool dummy_parameter_entity_p(entity p)
{
  string pn = entity_name(p);
  string dummy = strstr(pn, DUMMY_PARAMETER_PREFIX);
  bool is_dummy = (pn==dummy);

  pips_debug(9, "pn=\"%s\", dummy=\"%s\"\n", pn, dummy);

  return is_dummy;
}


/* This is useful for the C language only */
entity MakeCompilationUnitEntity(const char* name)
{
  entity e = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME,name);

  pips_assert("name is a compilation unit name", compilation_unit_p(name));

  /* Normally, the storage must be rom but in order to store the list of entities
     declared with extern, we use the ram storage to put this list in ram_shared*/
  if(!storage_undefined_p(entity_storage(e)))
    free_storage(entity_storage(e));
  entity_storage(e) = make_storage(is_storage_rom, UU);

  if(!type_undefined_p(entity_type(e)))
    free_type(entity_type(e));
  entity_type(e) = make_type_functional(make_functional(NIL,make_type_unknown()));

  if(!value_undefined_p(entity_initial(e)))
    free_value(entity_initial(e));
  entity_initial(e) = make_value(is_value_code, make_code(NIL,strdup(""), make_sequence(NIL),NIL, make_language_c()));

  return e;
}

bool extern_entity_p(entity module, entity e)
{
  /* There are two cases for "extern"

     - The current module is a compilation unit and the entity is in
       the ram_shared list of the ram storage of the compilation unit.

     - The current module is a normal function and the entity has a
       global scope.
*/
  // Check if e belongs to module
  /* bool isbelong = true;
  list ld = entity_declarations(m);
  //ifdebug(1) {
    pips_assert("e is visible in module",gen_in_list_p(e,ld));
  //  pips_assert("module is a module or compilation unit",entity_module_p(m)||compilation_unit_entity_p(m));
    pips_assert("e is either variable or function", variable_entity_p(e),functional_entity_p(e));
  //}
  if(variable_entity_p(e))
  //{
    if (compilation_unit_entity_p(m){
   //   return(strstr(entity_name(e),
    //}
    //else
      {
	//return(strstr(entity_name(e),TOP_LEVEL_MODULE_NAME) != NULL);
      //}
   //}
  //else
    //return(static_module_name_p(e));
  */
  /* return ((compilation_unit_entity_p(module) && gen_in_list_p(e,ram_shared(storage_ram(entity_storage(module)))))
	  ||(!compilation_unit_entity_p(module) && (strstr(entity_name(e),TOP_LEVEL_MODULE_NAME) != NULL)));
  */
    return ((compilation_unit_entity_p(module) && gen_in_list_p(e,code_externs(value_code(entity_initial(module)))))
	  ||(!compilation_unit_entity_p(module) && (strstr(entity_name(e),TOP_LEVEL_MODULE_NAME) != NULL)));

}

bool explicit_extern_entity_p(entity module, entity e)
{
  /* There are two cases for "extern"

     - The current module is a compilation unit and the entity is in
       the ram_shared list of the ram storage of the compilation unit.

     - The current module is a normal function and the entity has a
       global scope: this is not an explicit extern declaration.
  */
    return compilation_unit_entity_p(module)
	     && gen_in_list_p(e,code_externs(value_code(entity_initial(module))));
}

string storage_to_string(storage s)
{
  string desc = string_undefined;

  if(storage_undefined_p(s))
    desc = "storage_undefined";
  else if(storage_return_p(s))
    desc = "return";
  else if(storage_ram_p(s))
    desc = "ram";
  else if(storage_formal_p(s))
    desc = "formal";
  else if(storage_rom_p(s))
    desc = "rom";
  else
    pips_internal_error("Unknown storage tag");

  return desc;
}

/* Find the enclosing module of an entity. If an entity is a module, return e.
 If the entity is a top-level entity, return e.*/
/* FI: I'm surprised this function does not exist already */
entity entity_to_module_entity(entity e)
{
  entity m = entity_undefined;

  if(top_level_entity_p(e))
    m = e;
  else if(entity_module_p(e))
    m = e;
  else {
    const char* mn = entity_module_name(e);
    m = module_name_to_entity(mn);
  }

  pips_assert("entity m is defined", !entity_undefined_p(m));
  pips_assert("entity m is a module or top_level_entity", entity_module_p(m)||top_level_entity_p(e) );

  return m;
}

void update_dummy_parameter(parameter p, entity ep)
{
  if(dummy_unknown_p(parameter_dummy(p))) {
    free_dummy(parameter_dummy(p));
    parameter_dummy(p) = make_dummy_identifier(ep);
  }
  else {
    dummy d = parameter_dummy(p);

    pips_debug(8, "Dummy identifier changed from \"\%s\" to \"\%s\"\n",
	       entity_name(dummy_identifier(d)), entity_name(ep));
    /* Note that free_entity(dummy_identifier(d)) should be performed... */
    dummy_identifier(d) = ep;
  }
}

/* Returns true when f has no parameters */
bool parameter_passing_mode_p(entity f, int tag)
{
  type ft = ultimate_type(entity_type(f));

  /* Calls thru pointers require syntax_application */
  pips_assert("call to a function", type_functional_p(ft));

  functional ftf = type_functional(ft);
  bool mode_p = true;


  if(!ENDP(functional_parameters(ftf))) {
    /* It is assumed that all parameters are passed the same way,
       either by valule or by reference */
    parameter p = PARAMETER(CAR(functional_parameters(ftf)));
    mode_p = (((int)mode_tag(parameter_mode(p)))==tag);
  }
  else {
    /* We are in trouble... because we have to call a higher-level
       function from the preprocessor library. */
    if(c_module_p(f))
      mode_p = (tag==is_mode_value);
    else
      mode_p = (tag==is_mode_reference);
  }
  return mode_p;
}
bool parameter_passing_by_value_p(entity f)
{
  return parameter_passing_mode_p(f, is_mode_value);
}

bool parameter_passing_by_reference_p(entity f)
{
  return parameter_passing_mode_p(f, is_mode_reference);
}

/* This function concatenate a package name and a local name to
   produce a global entity name.

   Previous comment: This function creates a fortran operator parameter, i.e. a zero
   dimension variable with an overloaded basic type.

   Moved from bootstrap.c
 */
char * AddPackageToName(p, n)
     string p, n;
{
  string ps;
  int l;

  l = strlen(p);
  ps = gen_strndup(p, l + 1 + strlen(n) +1);

  *(ps+l) = MODULE_SEP;
  *(ps+l+1) = '\0';
  strcat(ps, n);

  return(ps);
}

/* Returns the binary operator associated to a C update operator such as +=

   If the operator is unknown, an undefined entity is returned.
*/
entity update_operator_to_regular_operator(entity op)
{
  entity sop = entity_undefined;

  if(ENTITY_PLUS_UPDATE_P(op))
    sop = entity_intrinsic(PLUS_C_OPERATOR_NAME);
  else if(ENTITY_MINUS_UPDATE_P(op))
    sop = entity_intrinsic(MINUS_C_OPERATOR_NAME);
  else if(ENTITY_MULTIPLY_UPDATE_P(op))
    sop = entity_intrinsic(MULTIPLY_OPERATOR_NAME);
  else if(ENTITY_DIVIDE_UPDATE_P(op))
    sop = entity_intrinsic(DIVIDE_OPERATOR_NAME);
  else if(ENTITY_MODULO_UPDATE_P(op))
    sop = entity_intrinsic(C_MODULO_OPERATOR_NAME);
  else if(ENTITY_LEFT_SHIFT_UPDATE_P(op))
    sop = entity_intrinsic(LEFT_SHIFT_OPERATOR_NAME);
  else if(ENTITY_RIGHT_SHIFT_UPDATE_P(op))
    sop = entity_intrinsic(RIGHT_SHIFT_OPERATOR_NAME);
  else if(ENTITY_BITWISE_AND_UPDATE_P(op))
    sop = entity_intrinsic(BITWISE_AND_OPERATOR_NAME);
  else if(ENTITY_BITWISE_XOR_UPDATE_P(op))
    sop = entity_intrinsic(BITWISE_XOR_OPERATOR_NAME);
  else if(ENTITY_BITWISE_OR_UPDATE_P(op))
    sop = entity_intrinsic(BITWISE_OR_OPERATOR_NAME);

  return sop;
}






/**
 * checks if an entity is an equivalent
 *
 * @param e entity to check
 *
 * @return true if entity is an equivalent
 */
bool entity_equivalence_p(entity e)
{
  return storage_ram_p(entity_storage(e))
    && !ENDP( ram_shared(storage_ram(entity_storage(e)) ));
}

/**
 * compare entity names
 *
 * @param e1 first entity
 * @param e2 second entity
 *
 * @return true if e1 and e2 have the same name
 */
bool same_entity_name_p(entity e1, entity e2)
{
    return same_string_p(entity_name(e1), entity_name(e2));
}

/**
 * look for @a ent in @a ent_l
 *
 * @param ent entity to find
 * @param ent_l list to scan
 *
 * @return true if @a ent belongs to @a ent_l
 */
bool entity_in_list_p(entity ent, list ent_l)
{
  return !gen_chunk_undefined_p(gen_find_eq(ent,ent_l));
}

/* returns l1 after elements of l2 but not of l1 have been appended to l1. */
/* l2 is freed */
/**
 * append all elements of l2 not in l1 to l1 and free l2
 *
 * @param l1 list to append entities to
 * @param l2 list from which the new entities come
 *
 * @return @a l1 with extra new entities appended
 */
list concat_new_entities(list l1, list l2)
{
  list new_l2 = NIL;
  set s = set_make(set_pointer);
  set_assign_list(s, l1);
  FOREACH(ENTITY,e,l2) {
    if( ! set_belong_p(s,e) )
      new_l2=CONS(ENTITY,e,new_l2);
  }
  gen_free_list(l2);
  set_free(s);
  return gen_nconc(l1, gen_nreverse(new_l2));
}

/**
 * check if e is used to declare one of the entities in entity list ldecl
 *
 * @param e entity to check
 * @param ldecl list of entities whose declaration may use e
 *
 * @return @a true if e appears in one of the declaration in ldecl
 */
bool entity_used_in_declarations_p(entity e, list ldecl)
{
  bool found_p = false;

  FOREACH(ENTITY, d, ldecl) {
    type dt = entity_type(d);
    list sel = type_supporting_entities(NIL, dt);

    if(gen_in_list_p(e, sel)) {
      pips_debug(8, "entity \"%s\" is used to declare entity \"%s\"\n",
		 entity_name(e), entity_name(d));
      found_p = true;
      gen_free_list(sel);
      break;
    }
    else {
      gen_free_list(sel);
    }
  }

  return found_p;
}

/**
 * check if e is used to declare one of the entities in entity list ldecl
 *
 * @param e entity to check
 * @param ldecl list of entities whose declaration may use e
 *
 * @return @a true if e appears in one of the declaration in ldecl
 */
bool type_used_in_type_declarations_p(entity e, list ldecl)
{
  bool found_p = false;

  FOREACH(ENTITY, d, ldecl) {
    /* The dummy declaration may be hidden in a struct or a union
       declaration. Maybe it could also be hidden in a function
       declaration. */
    type dt = entity_type(d);
    if(entity_struct_p(d) || entity_union_p(d) || type_functional_p(dt)) {
      list stl = type_supporting_types(dt);

      if(gen_in_list_p(e, stl)) {
	pips_debug(8, "entity \"%s\" is used to declare entity \"%s\"\n",
		   entity_name(e), entity_name(d));
	found_p = true;
	gen_free_list(stl);
	break;
      }
      else {
	gen_free_list(stl);
      }
    }
  }

  return found_p;
}


/* Create a copy of an entity, with (almost) identical type, storage
   and initial value, but a slightly different name as entities are
   uniquely known by their names, and a different offset if the
   storage is ram.

   Entity e must be defined or the function core dumps.

   Depending on its storage, the new entity might have to be inserted
   in code_declarations and the memory allocation recomputed.

   Depending on the language, the new entity might have to be inserted
   in statement declarations. This is left up to the user of this function.

   @return the new entity.
*/
entity make_entity_copy(entity e)
{
  entity ne = entity_undefined;
  char * variable_name = strdup(entity_name(e));
  int number = 0;

  /* Find the first matching non-already existent variable name: */
  do {
    if (variable_name != NULL)
      /* Free the already allocated name in the previous iteration that
	 was conflicting: */
      free(variable_name);
    asprintf(&variable_name, "%s_%d", entity_name(e), number++);
  }
  while(gen_find_tabulated(variable_name, entity_domain)
    != entity_undefined);

  ne = make_entity(variable_name,
		   copy_type(entity_type(e)),
		   copy_storage(entity_storage(e)),
		   copy_value(entity_initial(e)));

  if(storage_ram_p(entity_storage(ne))) {
    /* We are in trouble. Up to now, we have created a static alias of
     * the variable e (it's a variable entity because of its
     * storage). Note that static aliases do not exist in C.
     */
    ram r = storage_ram(entity_storage(ne));
    entity m = ram_function(r);

    /* FI: It would be better to perorm the memory allocation right
       away, instead of waiting for a later core dump in chains or
       ricedg, but I'm in a hurry. */
    ram_offset(r) = UNKNOWN_RAM_OFFSET;

    AddEntityToDeclarations(ne, m);
  }

  return ne;
}

/* Create a copy of an entity, with (almost) identical type, storage
   and initial value if move_initialization_p is false, but with a slightly
   different name as entities are uniquely known by their names, and a
   different offset if the storage is ram (still to be done).

   Entity e must be defined or the function core dumps.

   Depending on its storage, the new entity might have to be inserted
   in code_declarations (done) and the memory allocation recomputed (not done).

   Depending on the language, the new entity might have to be inserted
   in statement declarations. This is left up to the user of this function.

   For C, name collisions with the compilation unit are not checked
   here. They are unlikely, but should be checked by the caller.

   @return the new entity.
*/
entity generic_make_entity_copy_with_new_name(entity e,
					      string global_new_name,
					      bool systematically_add_suffix,
					      bool move_initialization_p)
{
  entity ne = entity_undefined;
  char * variable_name = strdup(global_new_name);
  int number = 0;

  if (systematically_add_suffix)
    asprintf(&variable_name, "%s_%d", global_new_name, number++);

  /* Find the first matching non-already existent variable name: */
  while(gen_find_tabulated(variable_name, entity_domain)
    != entity_undefined)
  {
    if (variable_name != NULL)
      /* Free the already allocated name in the previous iteration that
	 was conflicting: */
      free(variable_name);
    asprintf(&variable_name, "%s_%d", global_new_name, number++);
  }

  //extended_integer_constant_expression_p(e)

  ne = make_entity(variable_name,
		   copy_type(entity_type(e)),
		   copy_storage(entity_storage(e)),
		   move_initialization_p? copy_value(entity_initial(e)) :
		   make_value_unknown()
		   );

  if(storage_ram_p(entity_storage(ne))) {
    /* We are in trouble. Up to now, we have created a static alias of
     * the variable e (it's a variable entity because of its
     * storage). Note that static aliases do not exist in C.
     */
    ram r = storage_ram(entity_storage(ne));
    entity m = ram_function(r);

    /* FI: It would be better to perform the memory allocation right
       away, instead of waiting for a later core dump in chains or
       ricedg, but I'm in a hurry. -> fixed, BC.
    */
    //ram_offset(r) = UNKNOWN_RAM_OFFSET;

    const char * module_name = entity_module_name(ne);
    entity a = FindEntity(module_name, DYNAMIC_AREA_LOCAL_NAME);
    type t = entity_basic_concrete_type(ne);
    basic b = type_variable_p(t) ? variable_basic(type_variable(t)) : basic_undefined;
    int offset = 0;
    if (basic_undefined_p(b)) /* I don't know if this can happen, and what we should do in such case. BC. */
      offset = UNKNOWN_RAM_OFFSET;
    else
      {
	if (c_module_p(module_name_to_entity(module_name)))
	  offset = (basic_tag(b)!=is_basic_overloaded)?
	    (add_C_variable_to_area(a, ne)):(0);
	else
	  offset = (basic_tag(b)!=is_basic_overloaded)?
	    (add_variable_to_area(a, ne)):(0);
      }

    ram_offset(r) = offset;

    AddEntityToDeclarations(ne, m);
  }
  /* manage field renaming */
  type et = ultimate_type(entity_type(ne));
  if(type_variable_p(et)) {
      basic b = variable_basic(type_variable(et));
      if(basic_derived_p(b)) {
          entity derived = basic_derived(b);
          type derived_type = entity_type(derived);
          if(type_struct_p(derived_type) || type_union_p(derived_type)) {
              const char sep = type_struct_p(derived_type) ? STRUCT_PREFIX_CHAR : UNION_PREFIX_CHAR;
              list of_entities = type_struct(derived_type);
              for(list iter = of_entities; !ENDP(iter); POP(iter)) {
                  entity *field = (entity*)REFCAR(iter);
                  char *new_field_name;
                  asprintf(&new_field_name,"%s" MODULE_SEP_STRING "%s%s", entity_module_name(ne) , 1+strchr(entity_name(derived), sep), strchr(entity_name(*field), MEMBER_SEP_CHAR));
                  *field = make_entity_copy_with_new_name(*field,new_field_name,move_initialization_p);
                  free(new_field_name);
              }
          }
      }
  }
  return ne;
}


entity make_entity_copy_with_new_name(entity e,
				      string global_new_name,
				      bool move_initialization_p)
{
  return generic_make_entity_copy_with_new_name(e,
						global_new_name,
						false,
						move_initialization_p);
}

entity make_entity_copy_with_new_name_and_suffix(entity e,
				      string global_new_name,
				      bool move_initialization_p)
{
  return generic_make_entity_copy_with_new_name(e,
						global_new_name,
						true,
						move_initialization_p);
}




/* FI: it is assumed that thread safe entities are invariant with
   respect to workspaces. Another mechanism will be needed if user
   variables updated within a critical section also are added to the
   thread safe variable set.

   Thread safe entities are supposed to be updated within critical
   sections. Hence their dependence arcs may be ignored during
   parallelization. There is not gaurantee that the semantics is
   unchanged, for isntance pointer values are likely to differ, but
   havoc should be avoided and the semantics of programs that are not
   dependent on pointer values should be preserved.

   For the time begin, the set is implemented as a list because very
   few libc hidden variables are added.
*/
static list thread_safe_entities = NIL;

void add_thread_safe_variable(entity v)
{
  if(gen_in_list_p(v, thread_safe_entities)) {
    /* This might happen when a workspace is closed and another one
       open or created within one session. */
    //pips_internal_error("Thread-safe entity \"%s\" redeclared", entity_name(v));
    ;
  }
  else {
    /* The package name of v could be checked... especially if package names are unified. */
    thread_safe_entities = gen_nconc(thread_safe_entities, CONS(ENTITY, v, NIL));
  }
}

bool thread_safe_variable_p(entity v)
{
  bool thread_safe_p = gen_in_list_p(v, thread_safe_entities);

  return thread_safe_p;
}

/* FI: hidden variables added to take into account the side effects in
   the libc. Without them, dead code elimination would remove calls to
   rand or malloc. However, there is no useful information to be
   computed about them. Except perhard, the number of frees wrt the
   number of malloc. Hence, they are not taken into account by the
   semantics analysis.

   This set may not be a superset of the set of thread-safe variables.
*/
static list abstract_state_entities = NIL;


void add_abstract_state_variable(entity v)
{
  if(gen_in_list_p(v, abstract_state_entities)) {
    /* This might happen when a workspace is closed and another one
       open or created within one session. */
    //pips_internal_error("Thread-safe entity \"%s\" redeclared", entity_name(v));
    ;
  }
  else {
    /* The package name of v could be checked... especially if package names are unified. */
    abstract_state_entities = gen_nconc(abstract_state_entities, CONS(ENTITY, v, NIL));
  }
}

bool abstract_state_variable_p(entity v)
{
  bool abstract_state_p = gen_in_list_p(v, abstract_state_entities);

  return abstract_state_p;
}

/* Make sure that an list is an homogeneous list of entities */
bool entities_p(list el)
{
  bool success_p = true;

  FOREACH(ENTITY, e, el) {
    if(!check_entity(e)) {
      success_p = false;
      break;
    }
  }
  return success_p;
}

entity operator_neutral_element(entity op)
{
    const char * en = entity_user_name(op);

    const char * one_neutral []= {
        MULTIPLY_OPERATOR_NAME,
        MULTIPLY_UPDATE_OPERATOR_NAME,
        DIVIDE_UPDATE_OPERATOR_NAME,
        MIN_OPERATOR_NAME,
        AND_OPERATOR_NAME,
        NULL
    };
    for(int i=0;one_neutral[i];i++)
        if(same_string_p(one_neutral[i],en)) return make_integer_constant_entity(1);

    const char * plus_inf_neutral[] = {
        MIN_OPERATOR_NAME ,
        BITWISE_AND_OPERATOR_NAME,
        BITWISE_AND_UPDATE_OPERATOR_NAME,
        NULL
    };
    for(int i=0;plus_inf_neutral[i];i++)
        if(same_string_p(plus_inf_neutral[i],en)) {
            pips_user_warning("assuming reduction on integer\n");
            return make_integer_constant_entity(UINT_MAX);
        }

    const char * minus_inf_neutral[] = {
        MAX_OPERATOR_NAME,
        NULL
    };
    for(int i=0;minus_inf_neutral[i];i++)
        if(same_string_p(minus_inf_neutral[i],en)){
            pips_user_warning("assuming reduction on integer\n");
            return make_integer_constant_entity(INT_MIN);
        }

    const char * zero_neutral [] ={
        PLUS_OPERATOR_NAME,
        MINUS_UPDATE_OPERATOR_NAME,
        PLUS_UPDATE_OPERATOR_NAME,
        BITWISE_OR_OPERATOR_NAME,
        BITWISE_OR_UPDATE_OPERATOR_NAME,
        LEFT_SHIFT_UPDATE_OPERATOR_NAME,
        RIGHT_SHIFT_UPDATE_OPERATOR_NAME,
        PLUS_C_OPERATOR_NAME,
        OR_OPERATOR_NAME,
        NULL
    };
    for(int i=0;zero_neutral[i];i++)
        if(same_string_p(zero_neutral[i],en)) return make_integer_constant_entity(0);

    return entity_undefined;
}


/* Test if we are allowed to commute operations

   @param[in] c is the operation call

   @return true if we can commute operations

   Note that floating point operations are commutative, but since they are
   not associative due to rounding error , in a chain of operations, we
   cannot commute them. Of course, we should test whether an operation is
   alone or not to see if we are in this case...
*/
bool
commutative_call_p(call c)
{
    entity op  = call_function(c);
    bool commut_p = false;
    if(ENTITY_PLUS_P(op)||ENTITY_MULTIPLY_P(op)||ENTITY_AND_P(op)||ENTITY_OR_P(op)
            || ENTITY_PLUS_C_P(op) )
    {
        basic b = basic_of_call(c,false,true);
        switch(basic_tag(b))
        {
            case is_basic_float:
            case is_basic_complex:
                if(!get_bool_property("RELAX_FLOAT_ASSOCIATIVITY"))
                    break;
            case is_basic_logical:
            case is_basic_overloaded:
            case is_basic_int:
            case is_basic_pointer:
                commut_p=true;
                break;
            default:
                pips_internal_error("unhandled case");
        }
        free_basic(b);
    }
    return commut_p;
}


/**
 * @brief build a list of expression from a list of entities
 * @return the list of expression
 * @param l_ent, the list of entities
 **/
list entities_to_expressions(list l_ent) {
  list l_exp = NIL;
  FOREACH(ENTITY,e,l_ent) {
    l_exp = CONS(EXPRESSION, entity_to_expression(e), l_exp);
  }
  l_exp = gen_nreverse(l_exp);
  return(l_exp);
}


entity find_enum_of_member(entity m)
{
  entity mod = entity_to_module_entity(m);
  list dl = code_declarations(value_code(entity_initial(mod)));
  list sdl = list_undefined;
  list fdl = list_undefined;

  if(compilation_unit_entity_p(mod)) {
    /* if m was declared in the compilation unit cu and used elsewhere, cu may not be parsed yet. */
    sdl = NIL;
  }
  else {
    /* Not good in general, but should work for compilation units... */
    /* No, it does not... */
    sdl = entity_declarations(mod);
  }
  fdl = gen_nconc(gen_copy_seq(dl), gen_copy_seq(sdl));

  entity ee = entity_undefined;

  ifdebug(8) {
    pips_debug(8, "Declarations for enclosing module \"\%s\": \"", entity_name(mod));
    print_entities(dl);
    //print_entities(sdl);
    fprintf(stderr, "\"\n");
  }

  FOREACH(ENTITY, e,fdl) {
    if(entity_enum_p(e)) {
      list ml = type_enum(entity_type(e));

      pips_debug(8, "Checking enum \"\%s\"\n", entity_name(e));

      if(gen_in_list_p((void *) m, ml)) {
	ee = e;
	break;
      }
      ifdebug(8) {
	if(entity_undefined_p(ee)) {
	  pips_debug(8, "Member \"\%s\" not found in enum \"\%s\"\n",
		     entity_name(m), entity_name(e));
	}
	else {
	  pips_debug(8, "Member \"\%s\" found in enum \"\%s\"\n",
		     entity_name(m), entity_name(e));
	}
      }
    }
  }

  pips_assert("enum entity is found", !entity_undefined_p(ee));
  gen_free_list(fdl);

  return ee;
}


/** Test if a module is in C */
bool c_module_p(entity m)
{
  bool c_p = false;
  value v = entity_initial(m);

  if(!value_undefined_p(v)) {
    if(value_intrinsic_p(v))
        return true;
    language l = code_language(value_code(v));
    c_p = language_c_p(l);
    /* Temporary fix for the too many make_unknown_language()... */
    if(language_unknown_p(l))
      pips_internal_error("language should not be unknown");
  }
  else
      pips_internal_error("language should not be unknown");

  return c_p;
}




/** Test if a module is in Fortran */
/* Could be better factored in with C case */
bool fortran_module_p(entity m)
{
  /* FI->FC: the code that follows breaks the validation of Hpfc?!? */
  bool fortran_p = false;
  value v = entity_initial(m);
  if(!value_undefined_p(v)) {
    if(value_intrinsic_p(v))
        return true;
    fortran_p = language_fortran_p(code_language(value_code(v)));
  }
  else {
    /* If this alternative did not exist, the source code should be
       moved to ri-util*/
      pips_internal_error("language should not be unknown");
  }
  return fortran_p;
}

typedef struct { list le, lr; } deux_listes;

static void make_uniq_reference_list(reference r, deux_listes * l)
{
  entity e = reference_variable(r);
  if (! (storage_rom_p(entity_storage(e)) &&
	 !(value_undefined_p(entity_initial(e))) &&
	 value_symbolic_p(entity_initial(e)) &&
	 type_functional_p(entity_type(e)))) {

    /* Add reference r only once */
    if (l->le ==NIL || !gen_in_list_p(e, l->le)) {
      l->le = CONS(ENTITY,e,  l->le);
      l->lr = CONS(REFERENCE,r,l->lr);
    }
  }
}
/* FI: this function has not yet been extended for C types!!! */
list extract_references_from_declarations(list decls)
{
  list arrays = NIL;
  deux_listes lref = { NIL, NIL };

  FOREACH(ENTITY,e,decls) {
    type t = entity_type(e);

    if (type_variable_p(t) && !ENDP(variable_dimensions(type_variable(t))))
      arrays = CONS(VARIABLE,type_variable(t), arrays);
  }

  FOREACH(VARIABLE,v,arrays)
  {
  list ldim = variable_dimensions(v);
  while (!ENDP(ldim))
    {
      dimension d = DIMENSION(CAR(ldim));
      gen_context_recurse(d, &lref, reference_domain, make_uniq_reference_list, gen_null);
      ldim=CDR(ldim);

    }
  }
  gen_free_list(lref.le);

  return(lref.lr);
}
Pbase entity_list_to_base(l)
list l;
{
    list l2 = gen_nreverse(gen_copy_seq(l));
    Pbase result = BASE_NULLE;
    FOREACH(ENTITY, e, l2)
    {
	Pbase new = (Pbase) vect_new((Variable) e, VALUE_ONE);
	new->succ = result;
	result = new;
    }

    gen_free_list(l2);
    return(result);
}
/**
 * @name declarations updater
 * @{ */

typedef struct {
    set entities;
    bool (*chunk_filter)(void*);
    bool (*entity_filter)(entity);
} get_referenced_entities_t;

// helper storing an entity according to filter @p p->entity_filter
static void do_get_referenced_entities_on_entity(
        entity e, get_referenced_entities_t *p) {
    if(p->entity_filter(e))
        set_add_element(p->entities,p->entities,e);
}

/**
 * helper looking in a reference @p r for referenced entities
 */
static void do_get_referenced_entities_on_reference(reference r, get_referenced_entities_t *p)
{
    if(p->chunk_filter(r)) {
        entity e = reference_variable(r);
        do_get_referenced_entities_on_entity(e,p);
    }
}

/**
 * helper looking in a call for referenced entities
 */
static void do_get_referenced_entities_on_call(call c, get_referenced_entities_t* p)
{
    if(p->chunk_filter(c)) {
        entity e = call_function(c);
        do_get_referenced_entities_on_entity(e,p);
    }
}

/**
 * helper looking in a loop for referenced entities
 */
static void do_get_referenced_entities_on_loop(loop l, get_referenced_entities_t* p)
{
    if(p->chunk_filter(l)) {
        entity e = loop_index(l);
        do_get_referenced_entities_on_entity(e,p);
    }
}


/**
 * helper looking in a list for referenced entities
 */
static
void do_get_referenced_entities_on_list(list l, get_referenced_entities_t *p)
{
    FOREACH(ENTITY,e,l)
        do_get_referenced_entities_on_entity(e,p);
}

/**
 * helper looking in a ram for referenced entities
 */
static void do_get_referenced_entities_on_ram(ram r, get_referenced_entities_t *p)
{
    if(p->chunk_filter(r))
        do_get_referenced_entities_on_list(ram_shared(r),p);
}

/**
 * helper looking in an area for referenced entities
 */
static void do_get_referenced_entities_on_area(area a, get_referenced_entities_t *p)
{
    if(p->chunk_filter(a))
        do_get_referenced_entities_on_list(area_layout(a),p);
}
/**
 * helper looking in a statement declaration for referenced entities
 */
static void do_get_referenced_entities_on_statement(statement s, get_referenced_entities_t *p)
{
    if(p->chunk_filter(s))
        do_get_referenced_entities_on_list(statement_declarations(s),p);
    else {
      /* you skip the declarations, but not the value / type inside */
      FOREACH(ENTITY,e,statement_declarations(s)) {
        set tmp = get_referenced_entities_filtered(entity_initial(e),p->chunk_filter,p->entity_filter);
        set_union(p->entities,p->entities,tmp);
        set_free(tmp);
      }
      FOREACH(ENTITY,e,statement_declarations(s)) {
        set tmp = get_referenced_entities_filtered(entity_type(e),p->chunk_filter,p->entity_filter);
        set_union(p->entities,p->entities,tmp);
        set_free(tmp);
      }
    }
}

/* Same as get_referenced_entities,
 * but will only consider entities that
 * fulfills @p entity_filter
 * and will only enter consider entities **directly** involved in object
 * matching @p chunk_filter
 * \/!\ if you strip out statements, it will not consider declared entities, but it will consider their initial value
 */
set get_referenced_entities_filtered(void *elem,
        bool (*chunk_filter)(void*), bool (*entity_filter)(entity))
{
    set referenced_entities = set_make(set_pointer);
    if(!gen_chunk_undefined_p(elem)) {
      get_referenced_entities_t p = {
        referenced_entities,
        chunk_filter,
        entity_filter
      };

      /* if elem is an entity it self, add it */
      if(INSTANCE_OF(entity,(gen_chunkp)elem)) {
        entity e = (entity)elem;
        if(!entity_module_p(e)) {
          if(chunk_filter(entity_type(e)))
            gen_context_multi_recurse(entity_type(e),&p,
                                      reference_domain,gen_true,do_get_referenced_entities_on_reference,
                                      call_domain,gen_true,do_get_referenced_entities_on_call,
                                      NULL);
          if(!value_undefined_p(entity_initial(e)) && // struct fields have undefined initial
              chunk_filter(entity_initial(e)))
            gen_context_multi_recurse(entity_initial(e),&p,
                                      call_domain,gen_true,do_get_referenced_entities_on_call,
                                      reference_domain,gen_true,do_get_referenced_entities_on_reference,
                                      area_domain,gen_true,do_get_referenced_entities_on_area,
                                      ram_domain,gen_true,do_get_referenced_entities_on_ram,
                                      NULL);
        }
      }
      else {
        /* gather entities from elem */
        gen_context_multi_recurse(elem,&p,
            loop_domain,gen_true,do_get_referenced_entities_on_loop,
            reference_domain,gen_true,do_get_referenced_entities_on_reference,
            call_domain,gen_true,do_get_referenced_entities_on_call,
            statement_domain,gen_true,do_get_referenced_entities_on_statement,
            ram_domain,gen_true,do_get_referenced_entities_on_ram,
            NULL);
      }

      /* gather all entities referenced by referenced entities */
      list ltmp = set_to_list(referenced_entities);
      FOREACH(ENTITY,e,ltmp) {
        if(e!=elem) {
          set tmp = get_referenced_entities_filtered(e,chunk_filter,entity_filter);
          set_union(referenced_entities,referenced_entities,tmp);
          set_free(tmp);
        }
      }
      gen_free_list(ltmp);

      /* not merged with earlier test to avoid infinite recursion */
      if(INSTANCE_OF(entity,(gen_chunkp)elem)) {
        entity e = (entity)elem;
        set_add_element(referenced_entities,referenced_entities,e);
      }
    }

    return referenced_entities;
}

/* Default entity filter for get_referenced_entities()
 *
 * It filters out constants and intrinsics
 *
 * It should have been named entity_neither_constant_nor_intrinsic_p()...
 */
bool entity_not_constant_or_intrinsic_p(entity e) {
    return !entity_constant_p(e) && ! intrinsic_entity_p(e);
}
/**
 * retrieves the set of entities used in  @p elem
 * beware that this entities may be formal parameters, functions etc
 * so consider filter this set depending on your need,
 * using get_referenced_entities_filtered 
 *
 * @param elem  element to check (any gen_recursifiable type is allowded)
 *
 * @return set of referenced entities
 */
set get_referenced_entities(void* elem)
{
    return get_referenced_entities_filtered(elem,(bool (*)(void*))gen_true,
            entity_not_constant_or_intrinsic_p);

}
/**
 * Check if a variable is local to a module
 */
bool entity_local_variable_p(entity var, entity module) {
  bool local = false;

  if(storage_ram_p(entity_storage(var))) {
    ram r = storage_ram(entity_storage(var));
    if(same_entity_p(module, ram_function(r))) {
      entity section = ram_section(r);
      if(same_string_p(entity_module_name(section),entity_user_name(module))) {
        local=true;
      }
    }
  } else if( storage_formal_p(entity_storage(var))) {
    /* it might be better to check the parameter passing mode itself,
       via the module type */
    bool fortran_p = fortran_module_p(module);
    bool scalar_p = entity_scalar_p(var);

    formal r = storage_formal(entity_storage(var));
    if(!fortran_p && scalar_p && same_entity_p(module, formal_function(r))) {
      local=true;
    }
  }
  pips_debug(4,"Looked if variable %s is local to function %s, result is %d\n",
             entity_name(var),entity_name(module), local);
  return local;
}


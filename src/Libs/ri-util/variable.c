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
/* Handling of entity as program variables
 * (see also entity.c for generic entities)
 */

// To have asprintf:
#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "linear.h"

#include "genC.h"
#include "misc.h"
#include "ri.h"

#include "ri-util.h"

#include "properties.h"
#include "preprocessor.h"

#include "parser_private.h"
#include "syntax.h"
bool
variable_entity_p(entity e)
{
  bool variable =
    (entity_storage(e)!=storage_undefined) &&
    storage_ram_p(entity_storage(e));

  return variable;
}

/* BEGIN_EOLE */ /* - please do not remove this line */
/* Lines between BEGIN_EOLE and END_EOLE tags are automatically included
   in the EOLE project (JZ - 11/98) */
bool
symbolic_constant_entity_p(entity e)
{
  bool symbolic_constant = entity_storage(e)!= storage_undefined
    && storage_rom_p(entity_storage(e))
    && entity_initial(e) != value_undefined
    && value_symbolic_p(entity_initial(e));

  return symbolic_constant;
}

/* END_EOLE */


/* Add a global variable e to the variable declarations of a module.

   It does nothing if e is already in the list.

   In the general case you should use AddLocalEntityToDeclarations() or
   AddEntityToCurrentModule() instead.

   Since in C, variables should be added to the statement declarations
   too, only use this function for special stuff like compilation unit and
   special area delarations in the module bootstrapping.
*/
void
AddEntityToDeclarations(entity e, entity module) {
    pips_assert("module is fine",entity_consistent_p(module));
    pips_assert("entity is fine",entity_consistent_p(e));
	/* Add the variable to the module declarations: */
	list l = code_declarations(EntityCode(module));
	/* Add the declaration only if not already here: */
	if (gen_chunk_undefined_p(gen_find_eq(e,l)))
		code_declarations(EntityCode(module))=CONS(ENTITY, e, l);
}

/**
 Add the variable entity e to the list of variables of the function
 module.

 For a C module, the variable is also added as local to the given
 statement s. A global variable to a module should be added to the global
 statement module (given by get_current_module_statement() for example.

 @param e variable entity to add
 @param module entity
 @param s statement where entity must be added. It can be
 statement_undefined in the case of a Fortran module
 */
void
AddLocalEntityToDeclarations(entity e, entity module, statement s) {
	/* SG: fix the entity storage if undefined
	 * it basically recompute the offset of a sclar variable
	 * I have not found how to do it for a variable size array, so I just dropped the case
	 */
	if( storage_undefined_p(entity_storage(e)) && entity_variable_p(e) && entity_scalar_p(e) )
	{
		entity dynamic_area = global_name_to_entity(module_local_name(module),DYNAMIC_AREA_LOCAL_NAME);
		entity_storage(e) = make_storage_ram(
				make_ram(module,
					dynamic_area,
					CurrentOffsetOfArea(dynamic_area, e),
					NIL)
				);
	}
	AddEntityToDeclarations(e, module);

	if (c_module_p(module)
			/* A compilation does not have statements */
			&& !compilation_unit_entity_p(module)) {
		/* In C the variable are local to a statement, so add : */
		pips_assert("Calling AddLocalEntityToDeclarations from c_module with valid statement", !statement_undefined_p(s) );
		list l = statement_declarations(s);
		if (gen_chunk_undefined_p(gen_find_eq(e,l)))
			statement_declarations(s) = CONS(ENTITY,e,l);
	}
}


/* Add a variable entity to the current module declarations. */
void
AddEntityToCurrentModule(entity e) {
  entity module_e = get_current_module_entity();
  /* There is no declaration local to a statement in Fortran: */
  statement module_s = c_module_p(module_e) ? get_current_module_statement()
    : statement_undefined;

  AddLocalEntityToDeclarations(e, module_e, module_s);
}

entity make_global_entity_from_local(entity local) {
    string seed = entity_local_name(local);
    int counter=0;
    entity new = entity_undefined;
    string eln= strdup(seed);
    while(!entity_undefined_p(FindEntity(TOP_LEVEL_MODULE_NAME,eln))) {
        asprintf(&eln,"%s%d",seed,counter++);
    }
    new = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME,eln);
    free(eln);
    entity_type(new)=copy_type(entity_type(local));
    entity_initial(new)=copy_value(entity_initial(local));
    entity a = global_name_to_entity(TOP_LEVEL_MODULE_NAME,DYNAMIC_AREA_LOCAL_NAME);
    entity f = local_name_to_top_level_entity(TOP_LEVEL_MODULE_NAME);
    entity_storage(new)=make_storage_ram(make_ram(f,a,add_any_variable_to_area(a,new,fortran_module_p(local)),NIL));
    return new;
}


/* entity make_scalar_entity(name, module_name, base)
 */
entity
make_scalar_entity(name, module_name, base)
string name;
string module_name;
basic base;
{
  string full_name;
  entity e, f, a;
  basic b = base;

  full_name =
    strdup(concatenate(module_name, MODULE_SEP_STRING, name, NULL));

  pips_debug(8, "name %s\n", full_name);

  message_assert("not already defined",
		 gen_find_tabulated(full_name, entity_domain)==entity_undefined);

  e = make_entity(full_name, type_undefined,
		  storage_undefined, value_undefined);

  entity_type(e) = (type) MakeTypeVariable(b, NIL);
  f = local_name_to_top_level_entity(module_name);
  a = global_name_to_entity(module_name, DYNAMIC_AREA_LOCAL_NAME);

  entity_storage(e) =
    make_storage(is_storage_ram,
		 make_ram(f, a,
			  (basic_tag(base)!=is_basic_overloaded)?
			  (add_variable_to_area(a, e)):(0),
			  NIL));

  /* FI: I would have expected is_value_unknown, especially with a RAM storage! */
  entity_initial(e) = make_value(is_value_constant,
				 MakeConstantLitteral());

  return(e);
}


/* -------------------------------------------------------------
 *
 * New Temporary Variables MANAGEMENT
 *
 */

static int
    unique_integer_number = 0,
    unique_float_number = 0,
    unique_logical_number = 0,
    unique_complex_number = 0,
    unique_string_number = 0;

void
reset_unique_variable_numbers()
{
  unique_integer_number=0;
  unique_float_number=0;
  unique_logical_number=0;
  unique_complex_number=0;
}

/* Default prefixes */
#define DEFAULT_INT_PREFIX 	"I_"
#define DEFAULT_FLOAT_PREFIX 	"F_"
#define DEFAULT_LOGICAL_PREFIX 	"L_"
#define DEFAULT_COMPLEX_PREFIX	"C_"
#define DEFAULT_STRING_PREFIX	"S_"
#define DEFAULT_POINTER_PREFIX	"P_"
#define DEFAULT_STRUCT_PREFIX	"ST_"
#define DEFAULT_UNION_PREFIX	"U_"
#define DEFAULT_ENUM_PREFIX	"E_"

/* Create a new scalar variable of type b in the given module.

   The variable name is constructed with "<prefix><number>" If the given
   prefix is the empty string, some standard prefixes are used, based on
   the type.

   In C this function is added to current module only.

   @return the variable entity.
*/
entity make_new_scalar_variable_with_prefix(string prefix,
					    entity module,
					    basic b)
{
  string module_name = module_local_name(module);
  entity e;
  char * variable_name = NULL;
  int number = 0;
  bool empty_prefix = (strlen(prefix) == 0);
  const string format = fortran_module_p(module)?"%s%d":"0" BLOCK_SEP_STRING "%s%d";

  /* Find the first matching non-already existent variable name: */
  do {
    if (variable_name != NULL)
      /* Free the already allocated name in the previous iteration that
	 was conflicting: */
      free(variable_name);

    if (empty_prefix) {
      /* Use a default type-dependent variable name since the programmer
	 gave none: */
		basic ub = basic_ultimate(b);
      switch(basic_tag(ub)) {
      case is_basic_int:
	asprintf(&variable_name,  format, DEFAULT_INT_PREFIX,
		unique_integer_number++);
	break;
      case is_basic_float:
	asprintf(&variable_name, format, DEFAULT_FLOAT_PREFIX,
		unique_float_number++);
	break;
      case is_basic_logical:
	asprintf(&variable_name, format, DEFAULT_LOGICAL_PREFIX,
		unique_logical_number++);
	break;
      case is_basic_complex:
	asprintf(&variable_name, format, DEFAULT_COMPLEX_PREFIX,
		unique_complex_number++);
	break;
      case is_basic_string:
	asprintf(&variable_name, format, DEFAULT_STRING_PREFIX,
		unique_string_number++);
	break;
      case is_basic_pointer:
	asprintf(&variable_name, format, DEFAULT_POINTER_PREFIX,
		unique_string_number++);
	break;
      case is_basic_derived: {
	entity de = basic_derived(b);
	type dt = ultimate_type(entity_type(de));

	if(type_struct_p(dt)) {
	   asprintf(&variable_name, format, DEFAULT_STRUCT_PREFIX,
		    unique_string_number++);
	}
	else if(type_union_p(dt)) {
	    asprintf(&variable_name, format, DEFAULT_UNION_PREFIX,
		     unique_string_number++);
	}
	if(type_enum_p(dt)) {
	   asprintf(&variable_name, format, DEFAULT_ENUM_PREFIX,
		    unique_string_number++);
	}
	else {
	  pips_internal_error("Not implemented for type tag: %d\n",
			      type_tag(dt));
	}
	break;
      }
      default:
	pips_internal_error("unknown basic tag: %d\n",
			    basic_tag(ub));
	break;
      }
    }
    else
      asprintf(&variable_name, format, prefix, number++);
  }
  while(gen_find_tabulated(concatenate(module_name,
				       MODULE_SEP_STRING,
				       variable_name,
				       NULL),
			   entity_domain) != entity_undefined);

  pips_debug(9, "var %s, tag %d\n", variable_name, basic_tag(b));

  e = make_scalar_entity(variable_name, module_name, b);
  // Add a global variable:
  AddEntityToCurrentModule(e);
  free(variable_name);

  return e;
}

entity make_new_scalar_variable(entity module, basic b)
{
  return make_new_scalar_variable_with_prefix("", module, b);
}


/* Make a new module integer variable of name X<d>.
 */
entity
make_new_module_variable(entity module,int d)
{

  static char name[ 64 ];
  string name1;
  entity ent1=entity_undefined;
  string full_name;
  static int num = 1;
  name[0] = 'X';
  if (d != 0) {
    (void) sprintf(&name[1],"%d",d);
    num = d;
  }
  else { (void) sprintf(&name[1],"%d",num);
  num++;}

  name1 = strdup(name);
  full_name=strdup(concatenate(module_local_name(module),
			       MODULE_SEP_STRING,
			       name1,
			       NULL));
  while ((ent1 = gen_find_tabulated(full_name,entity_domain))
	 != entity_undefined) {
    free(name1);
    free(full_name);
    name[0] = 'X';
    (void) sprintf(&name[1],"%d",num);
    num++;
    name1 = strdup(name);
    full_name=strdup(concatenate(module_local_name(module),
				 MODULE_SEP_STRING,
				 name1,
				 NULL));
  }
  ent1 = make_scalar_integer_entity(name1,
				    module_local_name(module));
  free(full_name);
  return ent1;
}


/* These globals variables count the number of temporary and auxiliary
 * entities. Each time such a variable is created, the corresponding
 * counter is incremented.
 *
 * FI: this must be wrong. A function to reset count_tmp and count_aux
 * is needed if tpips or wpips are to work in a consistent way!
 */
/* gcc complains that they are not used... but they are defined! */
static int count_tmp = 0;
static int count_aux = 0;


/* Make a new variable entity which name is one letter prefix + one
   incrementing number.

   The function name should be changed. Useless function according to
   previous ones ?

 * This entity is either a new temporary or a new auxiliary variable.
 * The parameter "kind" gives the kind of entity to produce.
 * "ba" gives the basic (ie the type) of the entity to create.
 *
 * The number of the temporaries is given by a global variable named
 * "count_tmp".
 * The number of the auxiliary variables is given by a global variable named
 * "count_aux".
 */
entity
make_new_entity(ba, kind)
basic ba;
int kind;
{
  extern list integer_entities,
    real_entities, logical_entities, complex_entities,
    double_entities, char_entities;

  entity new_ent, mod_ent;
  char prefix[4], *name, *num;
  int number = 0;
  entity dynamic_area;

  /* The first letter of the local name depends on the basic:
   *       int --> I
   *     real  --> F (float single precision)
   *    others --> O
   */
  switch(basic_tag(ba))
    {
    case is_basic_int: { (void) sprintf(prefix, "I"); break;}
    case is_basic_float:
      {
	if(basic_float(ba) == DOUBLE_PRECISION_SIZE)
	  (void) sprintf(prefix, "O");
	else
	  (void) sprintf(prefix, "F");
	break;
      }
    default: (void) sprintf(prefix, "O");
    }

  /* The three following letters are whether "TMP", for temporaries
   * or "AUX" for auxiliary variables.
   */
  switch(kind)
    {
    case TMP_ENT:
      {
	number = (++count_tmp);
	(void) sprintf(prefix+1, "TMP");
	break;
      }
    case AUX_ENT:
      {
	number = (++count_aux);
	(void) sprintf(prefix+1, "AUX");
	break;
      }
    default: user_error("make_new_entity", "Bad kind of entity: %d", kind);
    }

  mod_ent = get_current_module_entity();
  num = (char*) malloc(32);
  (void) sprintf(num, "%d", number);

  /* The first part of the full name is the concatenation of the define
   * constant ATOMIZER_MODULE_NAME and the local name of the module
   * entity.
   */
  /* ATOMIZER_MODULE_NAME discarded : it is a bug ! RK, 31/05/1994.
     name = strdup(concatenate(ATOMIZER_MODULE_NAME, entity_local_name(mod_ent),
     MODULE_SEP_STRING, prefix, num, (char *) NULL));
  */
  name = strdup(concatenate(entity_local_name(mod_ent),
			    MODULE_SEP_STRING, prefix, num, (char *) NULL));
  /*
    new_ent = make_entity(name,
    make_type(is_type_variable,
    make_variable(ba,
    NIL,NIL)),
    make_storage(is_storage_rom, UU),
    make_value(is_value_unknown, UU));
  */
  /* Create a true dynamic variable. RK, 31/05/1994 : */
  new_ent = make_entity(name,
			make_type(is_type_variable,
				  make_variable(ba,
						NIL,NIL)),
			storage_undefined,
			make_value(is_value_unknown, UU));
  dynamic_area = global_name_to_entity(module_local_name(mod_ent),
				       DYNAMIC_AREA_LOCAL_NAME);
  entity_storage(new_ent) = make_storage(is_storage_ram,
					 make_ram(mod_ent,
						  dynamic_area,
						  add_variable_to_area(dynamic_area, new_ent),
						  NIL));
  AddEntityToCurrentModule(new_ent);

  /* Is the following useless : */

  /* The new entity is stored in the list of entities of the same type. */
  switch(basic_tag(ba))
    {
    case is_basic_int:
      {
	integer_entities = CONS(ENTITY, new_ent, integer_entities);
	break;
      }
    case is_basic_float:
      {
	if(basic_float(ba) == DOUBLE_PRECISION_SIZE)
	  double_entities = CONS(ENTITY, new_ent, double_entities);
	else
	  real_entities = CONS(ENTITY, new_ent, real_entities);
	break;
      }
    case is_basic_logical:
      {
	logical_entities = CONS(ENTITY, new_ent, logical_entities);
	break;
      }
    case is_basic_complex:
      {
	complex_entities = CONS(ENTITY, new_ent, complex_entities);
	break;
      }
    case is_basic_string:
      {
	char_entities = CONS(ENTITY, new_ent, char_entities);
	break;
      }
    default:break;
    }

  return new_ent;
}


/* Looks for an entity which should be a scalar of the specified
   basic. If found, returns it, else one is created.

   If the entity is not a scalar, it aborts.
 */
entity
find_or_create_scalar_entity(name, module_name, base)
string name;
string module_name;
tag base;
{
  entity e = entity_undefined;
  string nom = concatenate(module_name, MODULE_SEP_STRING, name, NULL);

  if ((e = gen_find_tabulated(nom, entity_domain)) != entity_undefined)
    {
      pips_assert("find_or_create_scalar_entity",
		  (entity_scalar_p(e) && entity_basic_p(e, base)));

      return(e);
    }

  return(make_scalar_entity(name, module_name, MakeBasic(base)));
}


/* Looks for an entity of the specified
   basic. If found, returns it, else one is created.
 */
entity
find_or_create_typed_entity(
   string name,
   string module_name,
   tag base)
{
  entity e = entity_undefined;
  string nom = concatenate(module_name, MODULE_SEP_STRING, name, NULL);

  if ((e = gen_find_tabulated(nom, entity_domain)) != entity_undefined)
    {
      pips_assert("type is okay", entity_basic_p(e, base));

      return(e);
    }

  return(make_scalar_entity(name, module_name, MakeBasic(base)));
}


/* Create an integer variable of name "name" in module of name
   "module_name" */
entity
make_scalar_integer_entity(name, module_name)
char *name;
char *module_name;
{
  string full_name;
  entity e, f, a ;
  basic b ;

  debug(8,"make_scalar_integer_entity", "begin name=%s, module_name=%s\n",
	name, module_name);

  full_name = concatenate(module_name, MODULE_SEP_STRING, name, NULL);
  hash_warn_on_redefinition();
  e = make_entity(strdup(full_name),
		  type_undefined,
		  storage_undefined,
		  value_undefined);

  b = make_basic(is_basic_int, (void*) 4);

  entity_type(e) = (type) MakeTypeVariable(b, NIL);

  f = local_name_to_top_level_entity(module_name);
  a = global_name_to_entity(module_name, DYNAMIC_AREA_LOCAL_NAME);
  pips_assert("make_scalar_integer_entity", !entity_undefined_p(f) && !entity_undefined_p(a));

  entity_storage(e) = make_storage(is_storage_ram,
				   (make_ram(f, a,
					     add_any_variable_to_area(a, e,fortran_module_p(f)),
					     NIL)));

  entity_initial(e) = make_value(is_value_constant,
				 MakeConstantLitteral());

  debug(8,"make_scalar_integer_entity", "end\n");

  return(e);
}


/* The concrete type of e is a scalar type. The programmer cannot index
   this variable.

   Note: variable e may appear indexed somewhere in the PIPS internal
   representation if this is linked to some semantics.
*/
bool entity_scalar_p(entity e)
{
  type t = ultimate_type(entity_type(e));

  // hmmm... some hpfc validations end here:-(
  pips_assert("e is a variable", type_variable_p(t));

  return ENDP(variable_dimensions(type_variable(t)));
}

/* for variables (like I), not constants (like 1)!
 * use integer_constant_p() for constants
 *
 * The integer type may be signed or unsigned.
 */
bool entity_integer_scalar_p(entity e)
{
  return(entity_scalar_p(e) &&
	 basic_int_p(variable_basic(type_variable(ultimate_type(entity_type(e))))));
}

/* integer_scalar_entity_p() is obsolete; use entity_integer_scalar_p() */
bool integer_scalar_entity_p(entity e)
{
  type ct = ultimate_type(entity_type(e));
  return type_variable_p(entity_type(e)) &&
    basic_int_p(variable_basic(type_variable(ct))) &&
    variable_dimensions(type_variable(ct)) == NIL;
}

/* Any reference r such that reference_variable(r)==e accesses all
   bytes (or bits) allocated to variable e. In other words, any write
   of e is a kill. At least, if the reference indices are NIL in the
   case of pointers.

   The Newgen type of e must be "variable".

   FI: This function is much too naive because array dimensions may be
   hidden anywhere in a chain of typdef types. It might be much better
   to use type_depth() or to write a more specific function. Well, for
   some unknown reason, type_depth is not the answer.
 */
bool entity_atomic_reference_p(entity e)
{
  type t = entity_type(e);
  //variable vt = type_variable(t);
  type ut = ultimate_type(entity_type(e));
  variable uvt = type_variable(ut);
  bool atomic_p = FALSE;

  pips_assert("entity e is a variable", type_variable_p(ut));

  /* Kludge to work in case the dimension is part of the typedef or
     part of the type. */
  /* if(ENDP(variable_dimensions(uvt)) &&
     ENDP(variable_dimensions(vt))) {*/
  if(type_depth(t)==0) {
    /* The property is not true for overloaded, string, derived
       (typedef is impossible here) */
    basic ubt = variable_basic(uvt);
    atomic_p = basic_int_p(ubt) || basic_float_p(ubt) || basic_logical_p(ubt)
      || basic_complex_p(ubt) || basic_bit_p(ubt) || basic_pointer_p(ubt);
  }

  return atomic_p;
}

/**

   @return TRUE if the entity is a scalar but not a pointer, FALSE otherwise.
           (takes care of typedefs).
 */
bool entity_non_pointer_scalar_p(entity e)
{
  type ct = basic_concrete_type(entity_type(e));
  variable vt = type_variable(ct);
  bool atomic_p = FALSE;

  pips_assert("entity e is a variable", type_variable_p(ct));

  if(ENDP(variable_dimensions(vt)))
    {
      /* The property is not true for overloaded, string, derived
       */
      basic bt = variable_basic(vt);
      atomic_p = basic_int_p(bt) || basic_float_p(bt) || basic_logical_p(bt)
	|| basic_complex_p(bt) || basic_bit_p(bt);
    }

  free_type(ct);
  return atomic_p;
}




  /* Another semantics would be: is this reference r to e a kill for
     e? In general, this cannot be answered at the entity level only
     (see previous function) and the reference itself must be passed
     as an argument.

     FI: I'm not sure of the best location for this function in
     ri-util (no file reference.c).
 */

dimension entity_ith_dimension(entity e, int i)
{
  cons *pd;
  type t = entity_type(e);

  pips_assert("entity_ith_dimension", type_variable_p(t));

  pd = variable_dimensions(type_variable(t));

  while (pd != NIL && --i > 0)
    pd = CDR(pd);

  pips_assert("entity_ith_dimension", pd != NIL);

  return(DIMENSION(CAR(pd)));
}

/* boolean entity_unbounded_p(entity e)
 * input    : an array entity
 * output   : TRUE if the last dimension of the array is unbounded (*),
 *            FALSE otherwise.
 * modifies : nothing
 * comment  :
 */
boolean entity_unbounded_p(entity e)
{
  int nb_dim = NumberOfDimension(e);

  return(unbounded_dimension_p(entity_ith_dimension(e, nb_dim)));
}

/* boolean array_with_numerical_bounds_p(entity a)
 * input    : an array entity
 * output   : TRUE if all bounds of all dimensions are numerical
 *            FALSE otherwise (adjustable arrays, formal parameters).
 * modifies : nothing
 * comment  :
 */
bool array_with_numerical_bounds_p(entity a)
{
  int nb_dim = NumberOfDimension(a);
  int d;
  bool numerical_bounds_p = TRUE;

  for(d=1; d <= nb_dim && numerical_bounds_p; d++) {
    dimension dd = entity_ith_dimension(a, nb_dim);
    expression l = dimension_lower(dd);
    expression u = dimension_upper(dd);

    numerical_bounds_p = expression_with_constant_signed_integer_value_p(l)
      && expression_with_constant_signed_integer_value_p(u);
  }

  return numerical_bounds_p;
}



/* variable_entity_dimension(entity v): returns the dimension of variable v;
 * scalar have dimension 0
 */
int variable_entity_dimension(entity v)
{
  int d = 0;

  pips_assert("variable_entity_dimension", type_variable_p(entity_type(v)));

  MAPL(cd, {
    d++;
  },
       variable_dimensions(type_variable(entity_type(v))));

  return d;
}


void remove_variable_entity(entity v)
{
  /* FI: this is pretty dangerous as it may leave tons of dangling pointers;
   * I use it to correct early declarations of types functions as variables;
   * I assume that no pointers to v exist in statements because we are still
   * in the declaration phasis.
   *
   * Memory leaks: I do not know if NewGen free_entity() is recursive.
   */
  storage s = entity_storage(v);
  entity f = entity_undefined;
  code c = code_undefined;

  if(storage_undefined_p(s)) {
    string fn = entity_module_name(v);
    f = local_name_to_top_level_entity(fn);
  }
  else if(storage_ram_p(s)) {
    f = ram_function(storage_ram(s));
  }
  else if(storage_rom_p(s)) {
    f = entity_undefined;
  }
  else {
    pips_error("remove_variable_entity", "unexpected storage %d\n", storage_tag(s));
  }

  if(!entity_undefined_p(f)) {
    pips_assert("remove_variable_entity", entity_module_p(f));
    c = value_code(entity_initial(f));
    gen_remove(&code_declarations(c), v);
  }
  free_entity(v);
}

/* entity make_integer_constant_entity(int c)
 * make entity for integer constant c

 WARNING : the basic integer size is fixed to sizeof(_int) */
entity make_integer_constant_entity(_int c) {
  entity ce;
  /* 64 bits numbers are printed in decimal in 20 digits, so with - and \0
     32 is enough. */
  char num[32];
  string cn;

  sprintf(num, "%td", c);
  cn = concatenate(TOP_LEVEL_MODULE_NAME,MODULE_SEP_STRING,num,(char *)NULL);
  ce = gen_find_tabulated(cn,entity_domain);
  if (ce==entity_undefined) {		/* make entity for the constant c */
    functional cf =
      make_functional(NIL,
		      make_type(is_type_variable,
				make_variable(make_basic(is_basic_int, (void*)sizeof(int)),
					      NIL,NIL)));
    type ct = make_type(is_type_functional, cf);
    ce = make_entity(strdup(cn), ct, MakeStorageRom(),
		     make_value(is_value_constant,
				make_constant(is_constant_int, (void*)c)));
  }
  return(ce);
}

/*
 * These functions compute the current offset of the area a passed as
 * argument. The length of the variable v is also computed and then added
 * to a's offset. The initial offset is returned to the calling function.
 * v is added to a's layout if not already present. C and Fortran behaviours differ slightly.
 */

int add_variable_to_area(entity a, entity v)
{
  return(add_any_variable_to_area(a, v, TRUE));
}

int add_C_variable_to_area(entity a, entity v)
{
 return(add_any_variable_to_area(a, v, FALSE));
}

int add_any_variable_to_area(entity a, entity v, bool is_fortran_p)
{
  int OldOffset=-1;
  type ta = entity_type(a);
  area aa = type_area(ta);

  if(top_level_entity_p(a) && is_fortran_p ) {
    /* COMMONs are supposed to havethe same layout in each routine */
    pips_error("add_variable_to_area", "COMMONs should not be modified\n");
  }
  else {
    /* the local areas are StaticArea and DynamicArea in fortran */
    /* the areas are localStaticArea, localDynamicArea, moduleStaticArea, globalStaticArea in C*/
    int s = 0;
    OldOffset = area_size(aa);
    if(!SizeOfArray(v, &s)) {
      pips_internal_error("Varying size array \"%s\"\n", entity_name(v));
    }

    if(is_fortran_p)
      {
	area_layout(aa) = gen_nconc(area_layout(aa), CONS(ENTITY, v, NIL));
	area_size(aa) = OldOffset+s;
      }
    else
      {
	if(!gen_in_list_p(v, area_layout(aa)))
	  area_layout(aa) = gen_nconc(area_layout(aa), CONS(ENTITY, v, NIL));
	area_size(aa) = OldOffset+s;
      }
  }
  return(OldOffset);
}

bool formal_parameter_p(entity v)
{
    storage s = entity_storage(v);
    bool formal_p = storage_formal_p(s);

    return formal_p;
}


/* True if a variable is the pseudo-variable used to store value
   returned by a function: */
bool variable_return_p(entity v)
{
  storage s = entity_storage(v);
  bool return_p = storage_return_p(s);

  return return_p;
}


bool variable_is_a_module_formal_parameter_p(entity a_variable,
					     entity a_module)
{
  MAP(ENTITY, e,
  {
    storage s = entity_storage(e);
    if (e == a_variable) {
      if (storage_formal_p(s))
	/* Well, the variable is a formal parameter of the
	   module: */
	return TRUE;
      else
	/* The variable is in the declaration of the module
	   but is not a formal parameter: */
	return FALSE;
    }
  },
      code_declarations(value_code(entity_initial(a_module))));

  /* The variable is not in the declaration of the module: */
  return FALSE;
}

/* true if v is in a common. */
bool variable_in_common_p(entity v)
{
  return type_variable_p(entity_type(v)) &&
    storage_ram_p(entity_storage(v)) &&
    !entity_special_area_p(ram_section(storage_ram(entity_storage(v)))) ;
}

/* true if v appears in a SAVE statement, or in a DATA statement */
bool variable_static_p(entity v)
{
  return(type_variable_p(entity_type(v)) &&
	 storage_ram_p(entity_storage(v)) &&
	 static_area_p(ram_section(storage_ram(entity_storage(v)))));
}
bool variable_dynamic_p(entity v)
{
  return(type_variable_p(entity_type(v)) &&
	 storage_ram_p(entity_storage(v)) &&
	 dynamic_area_p(ram_section(storage_ram(entity_storage(v)))));
}

/* This test can only be applied to variables, not to functions, subroutines or
 * commons visible from a module.
 */
bool variable_in_module_p(entity v,
			  entity m)
{
  bool in_module_1 =
    strcmp(module_local_name(m), entity_module_name(v)) == 0;
  bool in_module_2 =
    entity_is_argument_p(v, code_declarations(value_code(entity_initial(m))));

  pips_assert ("both coherency",  in_module_1==in_module_2);

  return in_module_1;
}

bool variable_in_list_p(entity e, list l)
{
  bool is_in_list = FALSE;
  for( ; (l != NIL) && (! is_in_list); l = CDR(l))
    if(same_entity_p(e, ENTITY(CAR(l))))
      is_in_list = TRUE;
  return(is_in_list);
}


/* Discard the decls_text string of the module code to make the
   prettyprinter ignoring the textual declaration and remake all from
   the declarations without touching the corresponding property
   (PRETTYPRINT_ALL_DECLARATIONS). RK, 31/05/1994. */
void discard_module_declaration_text(entity a_module)
{
  code c = entity_code(a_module);
  string s = code_decls_text(c);

  free(s);
  code_decls_text(c) = strdup("");
}

/* Returns a numbered entity the name of which is suffix + number,
 * the module of which is prefix. Used by some macros to return
 * dummy and primed variables for system of constraints.
 *
 * moved to ri-util from hpfc on BC's request. FC 08/09/95
 */
entity get_ith_dummy(string prefix, string suffix, int i)
{
  char buffer[100];
  assert(i>=1 && i<=7);
  (void) sprintf(buffer, "%s%d", suffix, i);

  return find_or_create_scalar_entity(buffer, prefix, is_basic_int);
}


expression generate_string_for_alternate_return_argument(string i)
{
  expression e = expression_undefined;
  char buffer[9];

  pips_assert("A label cannot be more than 5 character long", strlen(i)<=5);
  buffer[0]='"';
  buffer[1]='*';
  buffer[2]=0;

  strcat(&buffer[0], i);

  buffer[strlen(i)+2]='"';
  buffer[strlen(i)+3]=0;

  e = MakeCharacterConstantExpression(strdup(buffer));

  return e;
}

/* * (star) used as formal label parameter is replaced by a string
   variable as suggested by Fabien Coelho. Its storage and initial value
   are lated initialized by MakeFormalParameter(). */
entity generate_pseudo_formal_variable_for_formal_label(string p, int l)
{
  entity fs = entity_undefined;
  string lsp = get_string_property("PARSER_FORMAL_LABEL_SUBSTITUTE_PREFIX");
  /* string lsp = "FORMAL_RETURN_LABEL_"; */
  /* let's assume that there are fewer than 999 formal label arguments */
  char buffer[4];
  string sn = &buffer[0];
  string full_name = string_undefined;

  pips_assert("No more than 999 alternate returns", l<999);

  sprintf(buffer, "%d", l);

  /* Generate a variable of type CHARACTER*(*). See gram.y,
     "lg_fortran_type:". It is postponed to MakeFormalParameter */
  full_name = strdup(concatenate(p, MODULE_SEP_STRING, lsp, sn, NULL));
  if((fs=gen_find_tabulated(full_name, entity_domain))==entity_undefined) {
    fs = make_entity(full_name,
		     type_undefined,
		     storage_undefined,
		     value_undefined);
  }
  else {
    /* fs may already exists if a ParserError occured or if an edit of the
       source file occured */
    free(full_name);
    full_name = string_undefined;

    /* Not so sure because CleanUpEntities() is called later, not
       before. This function is cvalled by the parser before the module
       declaration rule is reduced. */
    /*
    pips_assert("The type, storage and value are undefined\n",
		type_undefined_p(entity_type(fs))
		&& storage_undefined_p(entity_storage(fs))
		&& value_undefined_p(entity_initial(fs)));
    */
    /* Too bad for the memory leaks: they should not occur frequently */
    entity_type(fs) = type_undefined;
    entity_storage(fs) = storage_undefined;
    entity_initial(fs) = value_undefined;
  }

  /* Too early because the current_module_entity is not yet fully defined. */
  /* AddEntityToDeclarations(fs, get_current_module_entity()); */

  pips_debug(8, "Generated replacement for formal return label: %s\n",
	     entity_name(fs));

  return fs;
}

bool formal_label_replacement_p(entity fp)
{
  bool replacement_p = FALSE;

  string fpn = entity_local_name(fp);
  string lsp = get_string_property("PARSER_FORMAL_LABEL_SUBSTITUTE_PREFIX");
  /* string lsp = "FORMAL_RETURN_LABEL_"; */

  replacement_p = (strstr(fpn, lsp)==fpn);

  return replacement_p;
}

/* Assumes that eap is a call */
bool actual_label_replacement_p(expression eap)
{
  bool replacement_p = FALSE;
  if (expression_call_p(eap))
    {
      string ls = entity_user_name(call_function(syntax_call(expression_syntax(eap))));
      string p = ls+1;

      replacement_p = (strlen(ls) >= 4
		       && *ls=='"' && *(ls+1)=='*' && *(ls+strlen(ls)-1)=='"');

      if(replacement_p) {
	for(p=ls+2; p<ls+strlen(ls)-1; p++) {
	  if(*p<'0'||*p>'9') {
	    replacement_p =FALSE;
	    break;
	  }
	}
      }
    }

  return replacement_p;
}

bool call_contains_alternate_returns_p(call c)
{
  bool contains_p = FALSE;

  MAP(EXPRESSION, arg, {
    if((contains_p = actual_label_replacement_p(arg)))
      break;
  }, call_arguments(c));

  return contains_p;
}

/*
 * create a new entity for a new index variable with a name similar to
 * the old index name, and with the same type, declare it in the
 * current module and allocate it.
 */
entity make_new_index_entity(entity old_index, string suffix)
{
  entity new_index;
  string old_name;
  char *new_name=NULL;

  old_name = entity_name(old_index);

  /* add a terminal p till a new name is found. */
  for (asprintf(&new_name, "%s%s", old_name, suffix);
       gen_find_tabulated(new_name, entity_domain)!=entity_undefined;

       old_name = new_name) {
    free(new_name);
    asprintf(&new_name, "%s%s", old_name, suffix);
  }

  // FI: copy_storage() cree de l'aliasing entre new_index et old_index
  // Is this the right place to fix the problem?
  new_index = make_entity(new_name,
			  copy_type(entity_type(old_index)),
			  storage_undefined,
			  copy_value(entity_initial(old_index)));
  AddEntityToCurrentModule(new_index);
  return(new_index);
}

bool implicit_c_variable_p(entity v)
{
  string vn = entity_user_name(v);


  //  return string_equal_p(vn, IMPLICIT_VARIABLE_NAME_1)
  //|| string_equal_p(vn, IMPLICIT_VARIABLE_NAME_2);

  return strcmp(vn, IMPLICIT_VARIABLE_NAME_1) == 0
    || strcmp(vn, IMPLICIT_VARIABLE_NAME_2) == 0;

}


/* Returns a copy of the initial expression of variable v. If v's
   inital value is a constants or a code block, it is converted to the
   corresponding expression.
*/
expression variable_initial_expression(entity v)
{
  value val = entity_initial(v);
  expression exp = expression_undefined;

  if (value_expression_p(val)) {
    exp = copy_expression(value_expression(val));
  }
  else if(value_constant_p(val)) {
    constant c = value_constant(val);
    if (constant_int_p(c)) {
      exp = int_to_expression(constant_int(c));
    }
    else {
      pips_internal_error("Not Yet Implemented.\n");
    }
  }
  else if(value_code_p(val)) {
    if(pointer_type_p(ultimate_type(entity_type(v)))) {
      list il = sequence_statements(code_initializations(value_code(val)));

      if(!ENDP(il)) {
	statement is = STATEMENT(CAR(il));
	instruction ii = statement_instruction(is);

	pips_assert("A pointer initialization is made of one instruction expression",
		    gen_length(il)==1 && instruction_expression(ii));

	exp = copy_expression(instruction_expression(ii));
      }
    }
  }
  else if(value_unknown_p(val)) {
    exp = expression_undefined;
  }
  else {
    pips_internal_error("Unexpected value tag %d.\n", value_tag(val));
  }

  return exp;
}

/* Check if a variable is initialized by itself as "int a = a;" is
   legal C code according to gcc. */
bool self_initialization_p(entity v)
{
  bool self_p = FALSE;

  expression e = variable_initial_expression(v);

  if(expression_undefined_p(e))
    self_p = FALSE;
  else {
    /* sd v referenced in e? */
    list lr = expression_to_reference_list(e, NIL);

    FOREACH(REFERENCE, r, lr) {
      entity rv = reference_variable(r);
      if(v==rv) {
	self_p = TRUE;
	break;
      }
    }
    gen_free_list(lr);
  }
  return self_p;
}

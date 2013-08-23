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
/* Handling of entity as program variables
 * (see also entity.c for generic entities)
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "linear.h"

#include "genC.h"
#include "misc.h"
#include "ri.h"

#include "ri-util.h"

#include "properties.h"

#include "parser_private.h"
#include "syntax.h"
#include "resources.h"

/* Check that "name" can be used as a new variable name in module
   "in_module". Should work for C and for Fortran. Apparently, should
   work whether name is already a global name or not, hence the
   derivation of user_name

   Of course, not really debugged for Fortran:-(.
.*/
static bool unique_entity_name_p(const char * name, entity in_module)
{
    /* first recover a user_name from global_name */
    const char *user_name=
      strchr(name,BLOCK_SEP_CHAR)?global_name_to_user_name(name):name;
    /* first check in entity declaration, where all entities are added
     * At least AddEntityToDeclarations keep this information up to date
     */
    FOREACH(ENTITY,e,entity_declarations(in_module))
    {
        if(same_string_p(entity_user_name(e),user_name))
            return false;
    }
    /* everything seems ok, do a last check with gen_fin_tabulated */
    if(strstr(name,MODULE_SEP_STRING))
        return gen_chunk_undefined_p(gen_find_tabulated(name,entity_domain));
    else
        return gen_chunk_undefined_p(gen_find_tabulated(concatenate(module_local_name(in_module), MODULE_SEP_STRING,name,NULL),entity_domain));
}

/* See also macro entity_variable_p()... */
bool variable_entity_p(entity e)
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
    /* Add the variable to the module declarations */
    list l = 
        code_declarations(EntityCode(module));
    /* Add the declaration only if not already here: */
    if (gen_chunk_undefined_p(gen_find_eq(e,l)))
        code_declarations(EntityCode(module))=CONS(ENTITY, e, l);
}

void
RemoveLocalEntityFromDeclarations(entity e, entity module, statement s)
{
    if(!ENDP(entity_declarations(module)))
        gen_remove(&entity_declarations(module), e);
    if(!statement_undefined_p(s))
    {
        if(!ENDP(statement_declarations(s)))
            gen_remove(&statement_declarations(s), e);
        if(statement_block_p(s))
        {
            // iterate over a copy because FOREACH does not
            // support inplace modification
            list theblock = gen_copy_seq(statement_block(s));
            FOREACH(STATEMENT,stat,theblock)
            {
                bool decl_stat = declaration_statement_p(stat);
                RemoveLocalEntityFromDeclarations(e,module,stat);
                /* this take care of removing useless declaration statements*/
                if(ENDP(statement_declarations(stat)) && decl_stat)
                {
                    gen_remove_once(&instruction_block(statement_instruction(s)),stat);
                    free_statement(stat);
                }
            }
            gen_free_list(theblock);
        }
    }

}

/* See the two user interfaces below */
static void
GenericAddLocalEntityToDeclarations(entity e, entity module, statement s,
				    bool add_declaration_statement_p) {
  /* SG: fix the entity storage if undefined
   * it basically recompute the offset of a scalar variable
   * I have not found how to do it for a variable size array, so I
   * just dropped the case -> a variable size array must be allocated
   * in a different area, STACK_AREA, where offsets are not computed
   */
  if( storage_undefined_p(entity_storage(e)) && entity_variable_p(e) )
    {
      entity dynamic_area = FindEntity(module_local_name(module),
						  DYNAMIC_AREA_LOCAL_NAME);
      int tmp;
      if(SizeOfArray(e,&tmp)) { // << CurrentOffsetOfArea fails if SizeOfArray is not computable
      entity_storage(e) = make_storage_ram(
					   make_ram(module,
						    dynamic_area,
						    CurrentOffsetOfArea(dynamic_area, e),
						    NIL)
					   );
      }
      else {
          pips_user_warning("Varying size for array \"%s\"\n", entity_name(e));
          pips_user_warning("Not yet supported properly by PIPS\n");
      }
    }

  /* Both in C and Fortran, all variables and useful entities are
     stored in code_declarations, in the symbol table. */
  AddEntityToDeclarations(e, module);

  /* In C the variables, but the formal parameters, are local to a
     statement */
  if (c_module_p(module)) {
    /* If undeclared in s, variable e is added in the
       statement_declarations field. */
      if(!statement_block_p(s))
          insert_statement(s,make_continue_statement(entity_empty_label()),true);
      pips_assert("add declarations to statement block",statement_block_p(s));

    list l = statement_declarations(s);
    pips_assert("Calling AddLocalEntityToDeclarations from c_module with valid statement", !statement_undefined_p(s) );

    /* The entity may have already been declared... This could be an
       assert but Serge seems to redeclare several times the same
       variables when performing inlining */
    if (gen_chunk_undefined_p(gen_find_eq(e,l))) {
      statement_declarations(s) = gen_nconc(l,CONS(ENTITY,e,NIL));

      /* The C prettyprinter is not based on code_declarations or
	 statement_declarations but on declaration statements, which
	 happen to be continue statment for the time being. */
      if(!declaration_statement_p(s) && add_declaration_statement_p) {
	/* To preserve the source layout, declarations are
	   statements */
	add_declaration_statement(s, e);
      }
    }
  }
}

/**
 Add the variable entity e to the list of variables of the function
 module.

 For a C module, the variable is also added as local to the given
 statement s. A global variable to a module should be added to the global
 statement module (given by get_current_module_statement() for
 example. The consistency of the internal representation is maintained.

 @param e variable entity to add
 @param module entity
 @param s statement where entity must be added. A new declaraton
 statement for e is added. It can be
 statement_undefined in the case of a Fortran module
 */
void AddLocalEntityToDeclarations(entity e, entity module, statement s) {
  GenericAddLocalEntityToDeclarations(e, module, s, true);
}

/**
 Add the variable entity e to the list of variables of the function
 module.

 For a C module, the variable is also added as local to the given
 statement s. A global variable to a module should be added to the global
 statement module (given by get_current_module_statement() for
 example. The consistency of the internal representation is not
 maintained, but this is useful for the controlizer.

 @param e variable entity to add
 @param module entity
 @param s statement where entity must be added. No new declaration
 ststement is added. It can be
 statement_undefined in the case of a Fortran module
 */
void AddLocalEntityToDeclarationsOnly(entity e, entity module, statement s) {
  GenericAddLocalEntityToDeclarations(e, module, s, false);
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

/* Add a variable entity to the current module declarations. */
void
AddEntityToCurrentModuleWithoutDeclaration(entity e) {
  entity module_e = get_current_module_entity();
  /* There is no declaration local to a statement in Fortran: */
  statement module_s = c_module_p(module_e) ? get_current_module_statement()
    : statement_undefined;

  AddLocalEntityToDeclarationsOnly(e, module_e, module_s);
}


entity make_global_entity_from_local(entity local) {
    const char* seed = entity_local_name(local);
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
    entity a = FindEntity(TOP_LEVEL_MODULE_NAME,DYNAMIC_AREA_LOCAL_NAME);
    entity f = local_name_to_top_level_entity(TOP_LEVEL_MODULE_NAME);
    entity_storage(new)=make_storage_ram(make_ram(f,a,add_any_variable_to_area(a,new,fortran_module_p(entity_to_module_entity(local))),NIL));
    return new;
}

/* If the parser has not (yet) encountered "stderr", a PIPS
   transformation or instrumentation phase may need "stderr" to
   generate AST code. This happens with array_bound_check at least. */
entity make_stderr_variable()
{
  /* It's a global variable */
  entity v = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME,
				STDERR_NAME);
  /* Unfortunately, I do not have an unknown basic to use... It
     should be a FILE * pointer... */
  basic b = make_basic_int(DEFAULT_INTEGER_TYPE_SIZE);
  entity f = FindEntity(TOP_LEVEL_MODULE_NAME,
				   TOP_LEVEL_MODULE_NAME);
  entity a = FindEntity(TOP_LEVEL_MODULE_NAME,
				   STATIC_AREA_LOCAL_NAME);

  pips_assert("f & a are defined", !entity_undefined_p(f)
	      && !entity_undefined_p(a));

  /* Its type is variable, scalar, */
  entity_type(v) = MakeTypeVariable(b, NIL);

  /* its storage must be the static area of top-level */
  entity_storage(v) = make_storage_ram(make_ram(f, a, UNKNOWN_RAM_OFFSET, NIL));

  /* Its initial value is unknown */
  entity_initial(v) = make_value_unknown();
  return v;
}

/* entity make_scalar_entity(name, module_name, base)
 */
entity
make_scalar_entity(name, module_name, base)
const char* name;
const char* module_name;
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
  a = FindEntity(module_name, DYNAMIC_AREA_LOCAL_NAME);

  int offset = 0;
  if (c_module_p(module_name_to_entity(module_name)))
    offset = (basic_tag(b)!=is_basic_overloaded)?
      (add_C_variable_to_area(a, e)):(0);
  else
    offset = (basic_tag(b)!=is_basic_overloaded)?
      (add_variable_to_area(a, e)):(0);

  entity_storage(e) =
    make_storage(is_storage_ram,
		 make_ram(f, a,
			  offset,
			  NIL));

  /* FI: I would have expected is_value_unknown, especially with a RAM storage! */
  entity_initial(e) = make_value_unknown();

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

/* Generate a new variable name from a seed name to a module

   @param seed_name is the main name of the variable name to generate

   @param prefix is the prefix to prepend to the variable name

   @param suffix is the suffix to append to the variable name

   @param module is the entity of the module the variable will belong to

   Is there is already a variable with this name, a new one is tried with
   some numerical suffixes.

   @return the entity of the new variable. Its fields are to be filled
   later before use.
*/
entity
generate_variable_with_unique_name_to_module(const char * seed_name,
					     const char * prefix,
					     const char * suffix,
					     entity module) {
  const char* format = fortran_module_p(module) ?
    "%s "MODULE_SEP_STRING "%s%s%s":
    "%s "MODULE_SEP_STRING "0" BLOCK_SEP_STRING "%s%s%s";

  const char* format_num = fortran_module_p(module) ?
    "%s "MODULE_SEP_STRING "%s%s%s%d":
    "%s "MODULE_SEP_STRING "0" BLOCK_SEP_STRING "%s%s%s%d";

  const char* module_name = module_local_name(module);
  string variable_name;
  int number = 0;

  /* First try a basic name without numeric suffix: */
  asprintf(&variable_name, format, module_name, prefix, seed_name, suffix);
  while(!unique_entity_name_p(variable_name,module))
  {
    /* Free the old name since it was already used: */
    free(variable_name);
    /* And try a new one with a number suffix: */
    asprintf(&variable_name, format_num, module_name, prefix, seed_name, suffix, number++);
  };

  entity e = make_entity(variable_name,
			 type_undefined,
			 storage_undefined,
			 value_undefined);
  return e;
}


/* clone a variable with a new name.

   The new variable is added in the different declaration lists.

   @param insert_p If true, for C code, a new declaration statement is
   inserted in "declaration_statement" to maintain the internal
   representation consistency. Else, no new declaration statement is
   inserted and the internal representation is no longer consistent
   for C code.

   @return the new cloned entity

   See useful interface below

*/
entity generic_clone_variable_with_unique_name(entity old_variable,
					       statement declaration_statement,
					       string prefix,
					       string suffix,
					       entity module,
					       bool insert_p) {
  const char * seed_name = entity_user_name(old_variable);
  entity new_variable = generate_variable_with_unique_name_to_module(seed_name,
								     prefix,
								     suffix,
								     module);

  /* Clone the attributes of the old variable into the new one: */
  entity_type(new_variable) = copy_type(entity_type(old_variable));
  entity_storage(new_variable) = copy_storage(entity_storage(old_variable));
  entity_initial(new_variable) = copy_value(entity_initial(old_variable));

  if(insert_p)
    AddLocalEntityToDeclarations(new_variable, module, declaration_statement);
  else
    AddLocalEntityToDeclarationsOnly(new_variable, module, declaration_statement);

  return new_variable;
}

/* Clone a variable with a new user name.

   @param old_variable is the variable to clone

   @param declaration_statement is the enclosing sequence (block)
   defining the scope where the new variable is visible. It must be a
   statement of kind sequence.

   @param prefix is the prefix to prepend to the variable name

   @param suffix is the suffix to append to the variable name

   @param module is the entity of the module the variable will belong to

   Is there is already a variable with this name, new names are tried with
   numerical suffixes.

   @return the entity of the new variable. Its fields are copies from the
   old one. That means that there may be some aliasing since the old one
   and the new one have the same offset (in the sense of IR RAM) and
   that use-def chains and dependence graphs are going to be wrong.

   The clone variable is added to the declaration list of
   "statement_declaration" and to the code declarations of module
   "module". A new declaration statement is inserted in the sequence
   of "declaration_statement". This maintains the consistency of PIPS
   internal representation for C and Fortran code.
 */
entity clone_variable_with_unique_name(entity old_variable,
				       statement declaration_statement,
				       string prefix,
				       string suffix,
				       entity module) {
  return generic_clone_variable_with_unique_name( old_variable,
						  declaration_statement,
						  prefix,
						  suffix,
						  module,
						  true);
}

/* Create a new scalar variable of type b in the given module.

   The variable name is constructed with "<prefix><number>" If the given
   prefix is the empty string, some standard prefixes are used, based on
   the type.

   In Fortran, the prefix is forced to upper case to be consistent
   with PIPS Fortran internal representation. All the default prefixes
   are assumed to be uppercase strings.

   In C this function is added to current module only.

   @return the variable entity.

   It is not clear why the default prefix is (re)computed in the repeat
   until loop rather than before entering it.
*/
entity make_new_scalar_variable_with_prefix(const char* prefix,
					    entity module,
					    basic b)
{
  const char* module_name = module_local_name(module);
  string ep = strdup(prefix);
  entity e;
  char * variable_name = NULL;
  int number = 0;
  bool empty_prefix = (strlen(prefix) == 0);
  const string format = fortran_module_p(module)?"%s%d":"0" BLOCK_SEP_STRING "%s%d";
  ep = fortran_module_p(module)? strupper(ep,ep) : ep;

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
	entity de = basic_derived(ub);
	type dt = ultimate_type(entity_type(de));

	if(type_struct_p(dt)) {
	   asprintf(&variable_name, format, DEFAULT_STRUCT_PREFIX,
		    unique_string_number++);
	}
	else if(type_union_p(dt)) {
	    asprintf(&variable_name, format, DEFAULT_UNION_PREFIX,
		     unique_string_number++);
	}
	else if(type_enum_p(dt)) {
	   asprintf(&variable_name, format, DEFAULT_ENUM_PREFIX,
		    unique_string_number++);
	}
	else {
	  pips_internal_error("Not implemented for type tag: %d",
			      type_tag(dt));
	}
	break;
      }
      default:
	pips_internal_error("unknown basic tag: %d",
			    basic_tag(ub));
	break;
      }
    }
    else
      asprintf(&variable_name, format, ep, number++);
  }
  while(!unique_entity_name_p(variable_name,module));

  pips_debug(9, "var %s, tag %d\n", variable_name, basic_tag(b));

  e = make_scalar_entity(variable_name, module_name, b);
  free(variable_name);
  free(ep);

  return e;
}

entity make_new_scalar_variable(entity module, basic b)
{
  return make_new_scalar_variable_with_prefix("", module, b);
}

/** Create an array entity
 * @param module_name is the name of the module part of the entity name
 * @param name is the user name of the entity
 * @param base is the basic type for the array
 * @param dimensions is the list of dimensions for the array
 */
static entity make_array_entity(const char* name,
                                const char* module_name,
                                basic base,
                                list dimensions) {
  string full_name;
  entity e, f, a;
  basic b = base;
  asprintf(&full_name,"%s"MODULE_SEP_STRING"%s",module_name,name);
  pips_debug(8, "name %s\n", full_name);
  int n =0;
  while(!entity_undefined_p(gen_find_tabulated(full_name, entity_domain))) {
      free(full_name);
      asprintf(&full_name,"%s"MODULE_SEP_STRING"%s%d",module_name,name,n++);
  }
  e = make_entity(full_name, type_undefined, storage_undefined, value_undefined);

  entity_type(e) = (type) MakeTypeVariable(b, dimensions);
  f = local_name_to_top_level_entity(module_name);
  a = FindEntity(module_name, DYNAMIC_AREA_LOCAL_NAME);
  entity_storage(e) =
    make_storage(is_storage_ram,
                 make_ram(f, a,
                          (basic_tag(base)!=is_basic_overloaded)?
                              (add_variable_to_area(a, e)):(0),
                              NIL));
  entity_initial(e) = make_value_unknown();
  return(e);
}


/* J'ai ameliore la fonction make_new_scalar_variable_with_prefix  */
/* afin de l'etendre  a des tableau   */

entity make_new_array_variable_with_prefix(const char* prefix, entity module,basic b,list dimensions)
{
  const char* module_name = module_local_name(module);
  entity e;
  e = make_array_entity(prefix, module_name, b, dimensions);
  return e;
}

entity make_new_array_variable(entity module,basic b,list dimensions) {
    return make_new_array_variable_with_prefix("", module,b,dimensions);
}

/*
	Create an pointer to an array simlar to `efrom' initialized with
	expression `from'
 */
entity make_temporary_pointer_to_array_entity_with_prefix(char *prefix,entity efrom, entity module,
					      expression from) {
  basic pointee = copy_basic(variable_basic(type_variable(entity_type(efrom))));
  list dims = gen_full_copy_list(variable_dimensions(type_variable(entity_type(efrom))));

  /* Make the pointer type */
  basic pointer = make_basic_pointer(make_type_variable(make_variable(pointee,
								      dims,
								      NIL)));
  /* Create the variable as a pointer */
  entity new = make_new_scalar_variable_with_prefix(prefix,
      module, pointer);
  /* Set its initial */
  entity_initial(new) = expression_undefined_p(from)?make_value_unknown():
    make_value_expression(make_expression(make_syntax_cast(make_cast(make_type_variable(make_variable(copy_basic(pointer),NIL,NIL)),copy_expression(from))),normalized_undefined));
  return new;
}

entity make_temporary_pointer_to_array_entity(entity efrom, expression from, entity module) {
    return make_temporary_pointer_to_array_entity_with_prefix("",efrom,module,from);
}





/* Make a new module integer variable of name X<d>.
 */
entity
make_new_module_variable(entity module,int d)
{

  string name;
  entity ent1=entity_undefined;
  static int num = 1;
  if (d != 0) {
    (void)asprintf(&name,"X%d",d);
    num = d;
  }
  else { (void) asprintf(&name,"X%d",num);
  num++;}

  while(!unique_entity_name_p(name,module))
  {
    string tmp = name;
    (void)asprintf(&name,"X%d",num);
    num++;
    free(tmp);
  }
  ent1 = make_scalar_integer_entity(name,
				    module_local_name(module));
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
entity make_new_entity(basic ba, int kind)
{
  extern list integer_entities,
    real_entities, logical_entities, complex_entities,
    double_entities, char_entities;

  entity new_ent, mod_ent;
  // prefix+1 (line#820) must hold 3 characters
  char prefix[5], *name;
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

  /* The first part of the full name is the concatenation of the define
   * constant ATOMIZER_MODULE_NAME and the local name of the module
   * entity.
   */
  /* ATOMIZER_MODULE_NAME discarded : it is a bug ! RK, 31/05/1994.
     name = strdup(concatenate(ATOMIZER_MODULE_NAME, entity_local_name(mod_ent),
     MODULE_SEP_STRING, prefix, num, (char *) NULL));
  */
  asprintf(&name,"%s" MODULE_SEP_STRING "%s%d",entity_local_name(mod_ent),prefix,number);
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
  dynamic_area = FindEntity(module_local_name(mod_ent),
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
entity find_or_create_scalar_entity(const char* name, const char* module_name, tag base)
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
   const char* module_name,
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
make_scalar_integer_entity(const char *name, const char *module_name)
{
    string full_name;
    entity e, f, a ;
    basic b ;

    pips_debug(8, "begin name=%s, module_name=%s\n", name, module_name);

    full_name = concatenate(module_name, MODULE_SEP_STRING, name, NULL);
    hash_warn_on_redefinition();
    e = make_entity(strdup(full_name),
		    type_undefined,
		    storage_undefined,
		    value_undefined);

    b = make_basic(is_basic_int, (void*) 4);

    entity_type(e) = (type) MakeTypeVariable(b, NIL);

    f = local_name_to_top_level_entity(module_name);
    a = FindEntity(module_name, DYNAMIC_AREA_LOCAL_NAME);
    pips_assert("make_scalar_integer_entity",
		!entity_undefined_p(f) && !entity_undefined_p(a));

    entity_storage(e)
      = make_storage(is_storage_ram,
		     (make_ram(f, a,
			       add_any_variable_to_area(a, e,fortran_module_p(f)),
			       NIL)));

    entity_initial(e) = make_value(is_value_constant,
				   make_constant_litteral());

    pips_debug(8, "end\n");

    return(e);
}


/* The concrete type of e is a scalar type. The programmer cannot index
   this variable.

   Note: variable e may appear indexed somewhere in the PIPS internal
   representation if this is linked to some semantics.
*/
bool entity_scalar_p(entity e)
{
  bool return_value = false;

  type t = ultimate_type(entity_type(e));
  if(type_variable_p(t)) {
    return_value = ENDP(variable_dimensions(type_variable(t)))
        && ! pointer_type_p(t);
  }
  return return_value;
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

/* return true if the entity is declared with the keyword static */
bool entity_static_variable_p(entity e) {
    storage s = entity_storage(e);
    return storage_ram_p(s) && static_area_p(ram_section(storage_ram(s)));
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
  bool atomic_p = false;

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

   @return true if the entity is a scalar but not a pointer, false otherwise.
           (takes care of typedefs).
 */
bool entity_non_pointer_scalar_p(entity e)
{
    bool atomic_p = false;
    type ct = entity_basic_concrete_type(e);
    if(type_variable_p(ct))  {
        variable vt = type_variable(ct);

        //pips_assert("entity e is a variable", type_variable_p(ct));
        //entity can be a functional in C too

        if(ENDP(variable_dimensions(vt)))
        {
            /* The property is not true for overloaded, string, derived
            */
            basic bt = variable_basic(vt);
            atomic_p = basic_int_p(bt) || basic_float_p(bt) || basic_logical_p(bt)
                || basic_complex_p(bt) || basic_bit_p(bt);
        }

    }
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

/* bool entity_unbounded_p(entity e)
 * input    : an array entity
 * output   : true if the last dimension of the array is unbounded (*),
 *            false otherwise.
 * modifies : nothing
 * comment  :
 */
bool entity_unbounded_p(entity e)
{
  int nb_dim = NumberOfDimension(e);

  return(unbounded_dimension_p(entity_ith_dimension(e, nb_dim)));
}

/* bool array_with_numerical_bounds_p(entity a)
 * input    : an array entity
 * output   : true if all bounds of all dimensions are numerical
 *            false otherwise (adjustable arrays, formal parameters).
 * modifies : nothing
 * comment  :
 */
bool array_with_numerical_bounds_p(entity a)
{
  int nb_dim = NumberOfDimension(a);
  int d;
  bool numerical_bounds_p = true;

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
 * scalar have dimension 0.
 *
 * This is not necessarily the dimensions because of typedefs. Another
 * function is able to collect dimensions hidden in typedefs, but also
 * via fields: see type_depth().
 */
int variable_entity_dimension(entity v)
{
  int d = 0;

  pips_assert("variable_entity_dimension", type_variable_p(entity_type(v)));

  FOREACH(DIMENSION, cd, variable_dimensions(type_variable(entity_type(v))))
    d++;

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
    const char* fn = entity_module_name(v);
    f = local_name_to_top_level_entity(fn);
  }
  else if(storage_ram_p(s)) {
    f = ram_function(storage_ram(s));
  }
  else if(storage_rom_p(s)) {
    f = entity_undefined;
  }
  else {
    pips_internal_error("unexpected storage %d", storage_tag(s));
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
    ce = make_entity(strdup(cn), ct, make_storage_rom(),
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
  return(add_any_variable_to_area(a, v, true));
}

int add_C_variable_to_area(entity a, entity v)
{
 return(add_any_variable_to_area(a, v, false));
}

int add_any_variable_to_area(entity a, entity v, bool is_fortran_p)
{
  int OldOffset=-1;
  type ta = entity_type(a);
  area aa = type_area(ta);

  if(top_level_entity_p(a) && is_fortran_p ) {
    /* COMMONs are supposed to have the same layout in each routine */
    pips_internal_error("COMMONs should not be modified");
  }
  else {
    /* the local areas are StaticArea and DynamicArea in fortran */
    /* the areas are localStaticArea, localDynamicArea,
       moduleStaticArea, globalStaticArea in C; but we also mange the
       stack for variable of dependent types, the heap area to model
       dynamic allocation and the formal area to model the formal
       context in C. */
    int s = 0;
    OldOffset = area_size(aa);
    /* FI: I have a (temporary?) problem with some stub functional
       variables generated by the points-to analysis. In fact, the
       should be declared as pointers to functions, not as
       functions. I also have problem with overloaded stubs... */
    type uet = ultimate_type(entity_type(v));
    if(!type_variable_p(uet)
       || overloaded_type_p(uet)
       || !SizeOfArray(v, &s)) {
      if(is_fortran_p)
        return DYNAMIC_RAM_OFFSET;
      else {
	if(!gen_in_list_p(v, area_layout(aa)))
	  area_layout(aa) = gen_nconc(area_layout(aa), CONS(ENTITY, v, NIL));
      }
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

int new_add_any_variable_to_area(entity a, entity v, bool is_fortran_p)
{
  int OldOffset=-1;
  type ta = entity_type(a);
  area aa = type_area(ta);

  if(top_level_entity_p(a) && is_fortran_p ) {
    /* COMMONs are supposed to havethe same layout in each routine */
    pips_internal_error("COMMONs should not be modified.");
  }
  else if(static_area_p(a) || dynamic_area_p(a)) {
    /* the local areas are StaticArea and DynamicArea in fortran */
    /* the areas are localStaticArea, localDynamicArea,
       moduleStaticArea, globalStaticArea in C*/
    int s = 0;
    OldOffset = area_size(aa);
    if(!SizeOfArray(v, &s)) {
      /* FI: should only happens with stack_area_p(a)... */
      return DYNAMIC_RAM_OFFSET;
    }

    if(is_fortran_p) {
      area_layout(aa) = gen_nconc(area_layout(aa), CONS(ENTITY, v, NIL));
      area_size(aa) = OldOffset+s;
    }
    else { /* C language */
      if(!gen_in_list_p(v, area_layout(aa)))
	area_layout(aa) = gen_nconc(area_layout(aa), CONS(ENTITY, v, NIL));
      area_size(aa) = OldOffset+s;
    }
  }
  else if(stack_area_p(a)) {
    /* By definition of the stack area, the offset is unknown because
       the size of the cariables is unknown statically. E.g. dependent
       types. This may happen in C99 or in Fortran 77 extensions. */
    if(!gen_in_list_p(v, area_layout(aa)))
      area_layout(aa) = gen_nconc(area_layout(aa), CONS(ENTITY, v, NIL));
  }
  else if(heap_area_p(a)) {
    /* FI: the points-to analysis is going to modify the symbol table
       under some properties... Maybe it would be better not to track
       buckets and abstract buckets declared in the heap? */
    pips_assert("Not possible for Fortran code", !is_fortran_p);
    if(!gen_in_list_p(v, area_layout(aa)))
      area_layout(aa) = gen_nconc(area_layout(aa), CONS(ENTITY, v, NIL));
  }
  else {
    pips_internal_error("Unexpected area kind: \"%s\".",
			entity_name(a));
  }
  return(OldOffset);
}

bool formal_parameter_p(entity v)
{
    storage s = entity_storage(v);
    bool formal_p = storage_formal_p(s);

    return formal_p;
}

/* Is v a global variable declared local to a C file such "static int i;" */
bool static_global_variable_p(entity v)
{
  // static global variables are decared in a compilation unit
  bool static_global_variable_p = compilation_unit_p(entity_module_name(v));

  return static_global_variable_p;
}

/* Is v a global variable such as "int i;"
 *
 * This is OK for C, but Fortran deals with commons.
 */
bool global_variable_p(entity v)
{
  // static global variables are decared in a compilation unit
  bool global_variable_p =
    (strcmp(entity_module_name(v), TOP_LEVEL_MODULE_NAME)==0);

  return global_variable_p;
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
  FOREACH(ENTITY, e, code_declarations(value_code(entity_initial(a_module))))
  {
    storage s = entity_storage(e);
    if (e == a_variable) {
      if (storage_formal_p(s))
	/* Well, the variable is a formal parameter of the
	   module: */
	return true;
      else
	/* The variable is in the declaration of the module
	   but is not a formal parameter: */
	return false;
    }
  }

  /* The variable is not in the declaration of the module: */
  return false;
}

/* true if v is in a common. */
bool variable_in_common_p(entity v)
{
  return type_variable_p(entity_type(v)) &&
    storage_ram_p(entity_storage(v)) &&
    !entity_special_area_p(ram_section(storage_ram(entity_storage(v)))) ;
}

/* true if v appears in a SAVE statement, or in a DATA statement, or
   is declared static i C. The size of v in bytes can  */
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

bool variable_stack_p(entity v)
{
  return(type_variable_p(entity_type(v)) &&
	 storage_ram_p(entity_storage(v)) &&
	 stack_area_p(ram_section(storage_ram(entity_storage(v)))));
}

bool variable_heap_p(entity v)
{
  return(type_variable_p(entity_type(v)) &&
	 storage_ram_p(entity_storage(v)) &&
	 heap_area_p(ram_section(storage_ram(entity_storage(v)))));
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
  // FI: should be a call to gen_in_list_p()
  bool is_in_list = false;
  for( ; (l != NIL) && (! is_in_list); l = CDR(l))
    if(same_entity_p(e, ENTITY(CAR(l))))
      is_in_list = true;
  return(is_in_list);
}

/* @return whether entity is a "volatile" variable
 *
 * See also entity_register_p()
 */
bool entity_volatile_variable_p(entity v)
{
  type t = entity_type(v);
  pips_assert("the entity must have type variable", type_variable_p(t));

  return volatile_variable_p(type_variable(t));
}

/* @return whether variable is a "volatile" variable
 *
 * See also entity_register_p()
 */
bool volatile_variable_p(variable v)
{
  bool volatile_p = false;

  // FI: no idea if volatile can he hidden in a typedef...
  list ql = variable_qualifiers(v);

  FOREACH(QUALIFIER, q, ql) {
    if(qualifier_volatile_p(q)) {
      volatile_p = true;
      break;
    }
  }
  return volatile_p;
}

/* The variable may turn out to be a function */
bool qualified_variable_p(entity v, unsigned int is_qualified)
{
  bool qualified_p = false;
  type t = entity_type(v);
  if(type_variable_p(t)) {
    // ifdebug(1) pips_assert("the entity must have type variable",
    // type_variable_p(t));
    // FI: no idea if volatile can he hidden in a typedef...
    variable vt = type_variable(t);
    list ql = variable_qualifiers(vt);

    FOREACH(QUALIFIER, q, ql) {
      if(qualifier_tag(q)==is_qualified) {
	qualified_p = true;
	break;
      }
    }
  }
  return qualified_p;
}

bool const_variable_p(entity v)
{
  return qualified_variable_p(v, is_qualifier_const);
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
entity generate_pseudo_formal_variable_for_formal_label(const char* p, int l)
{
  entity fs = entity_undefined;
  const char* lsp = get_string_property("PARSER_FORMAL_LABEL_SUBSTITUTE_PREFIX");
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
  bool replacement_p = false;

  const char* fpn = entity_local_name(fp);
  const char* lsp = get_string_property("PARSER_FORMAL_LABEL_SUBSTITUTE_PREFIX");
  /* string lsp = "FORMAL_RETURN_LABEL_"; */

  replacement_p = (strstr(fpn, lsp)==fpn);

  return replacement_p;
}

/* Assumes that eap is a call */
bool actual_label_replacement_p(expression eap)
{
  bool replacement_p = false;
  if (expression_call_p(eap))
    {
      const char * ls = entity_user_name(call_function(syntax_call(expression_syntax(eap))));
      const char * p = ls+1;

      replacement_p = (strlen(ls) >= 4
		       && *ls=='"' && *(ls+1)=='*' && *(ls+strlen(ls)-1)=='"');

      if(replacement_p) {
	for(p=ls+2; p<ls+strlen(ls)-1; p++) {
	  if(*p<'0'||*p>'9') {
	    replacement_p =false;
	    break;
	  }
	}
      }
    }

  return replacement_p;
}

bool call_contains_alternate_returns_p(call c)
{
  bool contains_p = false;

  FOREACH(EXPRESSION, arg, call_arguments(c))
    if((contains_p = actual_label_replacement_p(arg)))
      break;

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
  entity module = module_name_to_entity(entity_module_name(old_index));

  old_name = entity_name(old_index);

  /* add a terminal suffix till a new name is found. */
  for (asprintf(&new_name, "%s%s", old_name, suffix); !unique_entity_name_p(global_name_to_user_name(new_name),module); old_name = new_name) {
      char *tmp = new_name;
      asprintf(&new_name, "%s%s", old_name, suffix);
      free(tmp);
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
  const char * vn = entity_user_name(v);


  //  return string_equal_p(vn, IMPLICIT_VARIABLE_NAME_1)
  //|| string_equal_p(vn, IMPLICIT_VARIABLE_NAME_2);

  return strcmp(vn, IMPLICIT_VARIABLE_NAME_1) == 0
    || strcmp(vn, IMPLICIT_VARIABLE_NAME_2) == 0;

}


/* Returns a copy of the initial (i.e. initialization) expression of
   variable v. If v's inital value is a constants or a code block, it
   is converted to the corresponding expression.

   Could have been called entity_to_initialization_expression(), or
   entity_to_initial_expression(), but it only makes sense for
   variables.
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
      pips_internal_error("Not Yet Implemented.");
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
  else if(value_intrinsic_p(val)) {
    exp = expression_undefined;
  }
  else if(value_symbolic_p(val)) {
    symbolic s = value_symbolic(val);
    /* FI: not sure this is used in C; also, the constant field could be
       used as well. */
    exp = copy_expression(symbolic_expression(s));
  }
  else {
    pips_internal_error("Unexpected value tag %d.", value_tag(val));
  }

  return exp;
}

/* Check if a variable is initialized by itself as "int a = a;" is
   legal C code according to gcc. */
bool self_initialization_p(entity v)
{
  bool self_p = false;

  expression e = variable_initial_expression(v);

  if(expression_undefined_p(e))
    self_p = false;
  else {
    /* sd v referenced in e? */
    list lr = expression_to_reference_list(e, NIL);

    FOREACH(REFERENCE, r, lr) {
      entity rv = reference_variable(r);
      if(v==rv) {
	self_p = true;
	break;
      }
    }
    gen_free_list(lr);
  }
  return self_p;
}

/* FI: transferred from semantics (should be used for effect translation
   as well) */
bool same_scalar_location_p(entity e1, entity e2)
{
  storage st1 = entity_storage(e1);
  storage st2 = entity_storage(e2);
  entity s1 = entity_undefined;
  entity s2 = entity_undefined;
  ram r1 = ram_undefined;
  ram r2 = ram_undefined;
  bool same = false;

  /* e1 or e2 may be a formal parameter as shown by the benchmark m from CEA
   * and the call to SOURCE by the MAIN, parameter NPBF (FI, 13/1/93)
   *
   * I do not understand why I should return false since they actually have
   * the same location for this call site. However, there is no need for
   * a translate_global_value() since the usual formal/actual binding
   * must be enough.
   */
  /*
   * pips_assert("same_scalar_location_p", storage_ram_p(st1) && storage_ram_p(st2));
   */
  if(!(storage_ram_p(st1) && storage_ram_p(st2)))
    return false;

  r1 = storage_ram(entity_storage(e1));
  s1 = ram_section(r1);
  r2 = storage_ram(entity_storage(e2));
  s2 = ram_section(r2);

  if(s1 == s2) {
    if(ram_offset(r1) == ram_offset(r2))
      same = true;
    else {
      pips_debug(7,
		 "Different offsets %td for %s in section %s and %td for %s in section %s\n",
		 ram_offset(r1), entity_name(e1), entity_name(s1),
		 ram_offset(r2), entity_name(e2), entity_name(s2));
    }
  }
  else {
    pips_debug(7,
	       "Disjoint entitites %s in section %s and %s in section %s\n",
	       entity_name(e1), entity_name(s1),
	       entity_name(e2), entity_name(s2));
  }

  return same;
}

/* Assume that v is declared as a struct. Return the list of its fields.
 *
 * Apart from a possible assert, the same function should wor for a union.
*/
list struct_variable_to_fields(entity v)
{
  type c_t = entity_basic_concrete_type(v);
  list fl = derived_type_to_fields(c_t);
  return fl;
}

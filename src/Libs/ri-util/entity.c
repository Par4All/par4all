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
/* Functions closely related to the entity class, constructors, predicates,...
 */
// To have strndup():
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "linear.h"

#include "genC.h"
#include "misc.h"
#include "ri.h"

#include "ri-util.h"

void print_entities(list l)
{
  MAP(ENTITY, e, {
    fprintf(stderr, "%s ", entity_name(e));
  }, l);
}

bool unbounded_expression_p(expression e)
{
  syntax s = expression_syntax(e);
  if (syntax_call_p(s))
    {
      string n = entity_local_name(call_function(syntax_call(s)));
      if (same_string_p(n, UNBOUNDED_DIMENSION_NAME))
	return TRUE;
    }
  return FALSE;
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
static entity make_empty_module(string full_name,
				type r)
{
  string name = string_undefined;
  entity e = gen_find_tabulated(full_name, entity_domain);
  entity DynamicArea, StaticArea, StackArea, HeapArea;

  /* FC: added to allow reintrance in HPFC */
  if (e!=entity_undefined)
    {
      pips_user_warning("module %s already exists, returning it\n",
			full_name);
      return e;
    }

  pips_assert("undefined", e == entity_undefined);

  e = make_entity
    (strdup(full_name),
     make_type(is_type_functional,
	       make_functional(NIL, r)),
     MakeStorageRom(),
     make_value(is_value_code,
		make_code(NIL, strdup(""), make_sequence(NIL),NIL,
			  make_language_unknown())));

  name = module_local_name(e);
  DynamicArea = FindOrCreateEntity(name, DYNAMIC_AREA_LOCAL_NAME);
  entity_type(DynamicArea) = make_type_area(make_area(0, NIL));
  entity_storage(DynamicArea) = MakeStorageRom();
  entity_initial(DynamicArea) = MakeValueUnknown();
  AddEntityToDeclarations(DynamicArea, e);

  StaticArea = FindOrCreateEntity(name, STATIC_AREA_LOCAL_NAME);
  entity_type(StaticArea) = make_type_area(make_area(0, NIL));
  entity_storage(StaticArea) = MakeStorageRom();
  entity_initial(StaticArea) = MakeValueUnknown();
  AddEntityToDeclarations(StaticArea, e);

  StackArea = FindOrCreateEntity(name, STACK_AREA_LOCAL_NAME);
  entity_type(StackArea) = make_type_area(make_area(0, NIL));
  entity_storage(StackArea) = MakeStorageRom();
  entity_initial(StackArea) = MakeValueUnknown();
  AddEntityToDeclarations(StackArea, e);

  HeapArea = FindOrCreateEntity(name, HEAP_AREA_LOCAL_NAME);
  entity_type(HeapArea) = make_type_area(make_area(0, NIL));
  entity_storage(HeapArea) = MakeStorageRom();
  entity_initial(HeapArea) = MakeValueUnknown();
  AddEntityToDeclarations(HeapArea, e);

  return(e);
}

entity make_empty_program(string name)
{
  string full_name = concatenate(TOP_LEVEL_MODULE_NAME,
				 MODULE_SEP_STRING, MAIN_PREFIX, name, NULL);
  return make_empty_module(full_name, make_type(is_type_void, UU));
}

entity make_empty_subroutine(string name)
{
  string full_name = concatenate(TOP_LEVEL_MODULE_NAME,
				 MODULE_SEP_STRING, name, NULL);
  return make_empty_module(full_name, make_type(is_type_void, UU));
}

entity make_empty_function(string name, type r)
{
  string full_name = concatenate(TOP_LEVEL_MODULE_NAME,
				 MODULE_SEP_STRING, name, NULL);
  return make_empty_module(full_name, r);
}

entity make_empty_blockdata(string name)
{
  string full_name = concatenate(TOP_LEVEL_MODULE_NAME, MODULE_SEP_STRING,
				 BLOCKDATA_PREFIX, name, NULL);
  return make_empty_module(full_name, make_type(is_type_void, UU));
}


/* this function checks that e has an initial value code. if yes returns
it, otherwise aborts.  */

code EntityCode(entity e)
{
  value ve = entity_initial(e);
  pips_assert("EntityCode", value_tag(ve) == is_value_code);
  return(value_code(ve));
}

entity make_label(string strg)
{

  entity l = find_or_create_entity(strg);
  if( type_undefined_p(entity_type(l)) ) {
    entity_type(l) = (type) MakeTypeStatement();
    entity_storage(l) = (storage) MakeStorageRom();
    entity_initial(l) = make_value(is_value_constant,
				   MakeConstantLitteral());
  }
  return l;
}

/* This function returns a new label */
entity make_new_label(module_name)
char * module_name;
{
  /* FI: do labels have to be declared?*/
  /* FI: it's crazy; the name is usually derived from the entity
     by the caller and here the entity is retrieved from its name! */
  entity mod = local_name_to_top_level_entity(module_name);
  string strg = new_label_name(mod);
  return make_label(strg);

}

entity make_loop_label(int __attribute__ ((unused)) desired_number,
		char * module_name)
{
  entity e = make_new_label(module_name);
  return e;
}

static bool label_defined_in_statement = FALSE;
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
  label_defined_in_statement = FALSE;
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
/*const*/ string
entity_local_name(const entity e)
{
  string null_name = "null";
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

/* Returns the module local user name
 * SG: should return a const pointer
 */
/*const*/ string module_local_name(entity e)
{
  /* No difference between modules and other entities, except for prefixes */
  string name = local_name(entity_name(e));

  return (name
    + strspn(name, MAIN_PREFIX)
    + strspn(name, BLOCKDATA_PREFIX)
    + strspn(name, COMMON_PREFIX));
}

/* Returns a pointer towards the resource name. The resource name is
   the module local name: it may include the  */
string module_resource_name(entity e)
{
  string rn = entity_local_name(e);

  rn += strspn(rn, MAIN_PREFIX)
    + strspn(rn, BLOCKDATA_PREFIX)
    + strspn(rn, COMMON_PREFIX);

  return rn;
}

/* END_EOLE */

string label_local_name(entity e)
{
  string name = local_name(entity_name(e));
  return name+strlen(LABEL_PREFIX);
}

/* Return a name valid for sorting variables in vectors and constraint
   systems.

   @return the name or "TCST" if the entity is null.
*/
string entity_name_or_TCST(entity e)
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
    extern entity get_current_module_entity();
    entity m = get_current_module_entity();
    string s = string_undefined;

    pips_assert("entity_relative_name", !entity_undefined_p(m));

    s = (strcmp(module_local_name(m), entity_module_name(m)) == 0) ? 
	entity_local_name(e) : entity_name(e) ;

    return s;
}
*/

string entity_and_common_name(entity e)
{
  entity m = get_current_module_entity();
  string name ;
  pips_assert("some current entity", !entity_undefined_p(m));

  name = concatenate(entity_local_name(ram_section(storage_ram(entity_storage(e)))),
		     MODULE_SEP_STRING,entity_name(e),NIL);

  return name +strlen(COMMON_PREFIX);
}

bool entity_empty_label_p(entity e)
{
  return empty_label_p(entity_local_name(e));
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
    return FALSE;
  else {
    value v = entity_initial(e);
    return v!=value_undefined && value_code_p(v);
  }
}

bool entity_main_module_p(entity e)
{
  return entity_module_p(e)
    && (strspn(entity_local_name(e), MAIN_PREFIX)==1
	|| same_string_p(entity_local_name(e), "main"));
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

/* e is the field of a structure */
bool entity_field_p(entity e)
{
  string eln = entity_local_name(e);
  bool field_p = FALSE;

  if(*eln!='\'' && *eln!='"') {
    string pos = strrchr(eln, MEMBER_SEP_CHAR);

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
  string sn = strdup(entity_name(f)); /* structure name */
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
    pips_internal_error("Unexpected type tag %d\n", type_tag(st));

  pips_assert("st is a struct or union type",
	      type_struct_p(st) || type_union_p(st));

  /* FI: positions are counted from 1 on; do we want to subtract 1? */
  rank = gen_position((void *) f, fl);

  if(rank==0) {
    pips_internal_error("Field \"\%s\" is not part of its %s \"\%s\"\n",
			entity_name(f), type_struct_p(st)?"structure":"union" , entity_name(su));
  }

  return rank;
}

bool entity_enum_p(entity e)
{
  return type_enum_p(entity_type(e));
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

bool entity_struct_p(entity e)
{
  string ln = entity_local_name(e);
  bool struct_p = (*ln==STRUCT_PREFIX_CHAR)
    || (strstr(entity_name(e),DUMMY_STRUCT_PREFIX)!=NULL);
  return struct_p;
}

bool entity_union_p(entity e)
{
  string ln = entity_local_name(e);
  bool union_p = (*ln==UNION_PREFIX_CHAR)
    || (strstr(entity_name(e),DUMMY_UNION_PREFIX)!=NULL);
  return union_p;
}

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

string entity_module_name(entity e)
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

bool top_level_entity_p(entity e)
{
  /* This code is wrong because it only checks that entity_module_name(e)
   * is a prefix of TOP_LEVEL_MODULE_NAME. So it returns TRUE for variables
   * of a subroutine called TOP!
   *
   * The MODULE_SEP_STRING should be added to TOP_LEVEL_MODULE_NAME?
   *
   * To return FALSE quickly, TOP_LEVEL_MODULE_NAME should begin with
   * a special character never appearing in standard identifiers, for
   * instance * (star).
   */
  /*
    return(strncmp(TOP_LEVEL_MODULE_NAME,
    entity_name(e),
    strlen(entity_module_name(e))) == 0);
  */

  /* FI: It's late, I cannot think of anything better */
  /*
    int l = strlen(entity_module_name(e));
    bool top = FALSE;

    if(l==strlen(TOP_LEVEL_MODULE_NAME)) {
    top = (strncmp(TOP_LEVEL_MODULE_NAME,
    entity_name(e), l) == 0);
    }
  */

  bool top = (strcmp(TOP_LEVEL_MODULE_NAME, entity_module_name(e)) == 0);

  return top;
}

bool io_entity_p(entity e)
{
  return(strncmp(IO_EFFECTS_PACKAGE_NAME,
		 entity_name(e),
		 strlen(entity_module_name(e))) == 0);
}

bool rand_effects_entity_p(entity e)
{
  return(strncmp(RAND_EFFECTS_PACKAGE_NAME,
		 entity_name(e),
		 strlen(entity_module_name(e))) == 0);
}


bool intrinsic_entity_p(entity e)
{
  return (!value_undefined_p(entity_initial(e)) && value_intrinsic_p(entity_initial(e)));
}

bool symbolic_entity_p(entity e)
{
  return (!value_undefined_p(entity_initial(e)) && value_symbolic_p(entity_initial(e)));
}

/* FI: I do not understand this function name (see next one!). It seems to me
 * that any common or user function or user subroutine would
 * be returned.
 * FI: assert condition made stronger (18 December 1998)
 */
entity entity_intrinsic(string name)
{
  entity e = (entity) gen_find_tabulated(concatenate(TOP_LEVEL_MODULE_NAME,
						     MODULE_SEP_STRING,
						     name,
						     NULL),
					 entity_domain);

  pips_assert("entity_intrinsic", e != entity_undefined  && intrinsic_entity_p(e));
  return(e);
}



/* this function does not create an intrinsic function because they must
   all be created beforehand by the bootstrap phase (see
   bootstrap/bootstrap.c). */

entity CreateIntrinsic(string name)
{
  entity e = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME, name);
  pips_assert("entity is defined", e!=entity_undefined && intrinsic_entity_p(e));
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
    return(null_2-null_1);
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

/*   TRUE if var1 <= var2
 */
bool lexicographic_order_p(entity var1, entity var2)
{
  /*   TCST is before anything else
   */
  if ((Variable) var1==TCST) return(TRUE);
  if ((Variable) var2==TCST) return(FALSE);

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

/* return TRUE if the basic associated with entity e matchs the passed tag */
bool entity_basic_p(entity e,enum basic_utype basictag)
{
  return basic_tag(entity_basic(e)) == basictag;
}

/* Checks that el only contains entity*/
bool entity_list_p(list el)
{
  bool pure = TRUE;

  MAP(ENTITY, e,
      {
	static entity le = entity_undefined;
	pips_debug(10, "Entity e in list is \"%s\"\n", safe_entity_name(e));
	if(entity_domain_number(e)!=entity_domain) {
	  pips_debug(8, "Last entity le in list is \"%s\"\n", safe_entity_name(le));
	  pure = FALSE;
	  break;
	}
	le = e;
      }, el);
  return pure;
}

/* this function maps a local name, for instance P, to the corresponding
 * TOP-LEVEL entity, whose name is TOP-LEVEL:P. n is the local name.
 */

static string prefixes[] = {
    "",
    MAIN_PREFIX,
    BLOCKDATA_PREFIX,
    COMMON_PREFIX,
};

entity local_name_to_top_level_entity(string n)
{
  entity module = entity_undefined;
  int i;

  /* Extension with C: the scope of a module can be its compilation unit if this is
     a static module, not only TOP-LEVEL. */

  if (static_module_name_p(n)) {
    string cun = strdup(n);
    string sep = strchr(cun, FILE_SEP);
    //string ln = strchr(n, MODULE_SEP)+1;

    *(sep+1) = '\0';
    module = gen_find_tabulated(concatenate(cun, MODULE_SEP_STRING, n, NULL),entity_domain);
    free(cun);
  }
  else
    {
      for(i=0; i<4 && entity_undefined_p(module); i++)
	module = gen_find_tabulated(concatenate
				    (TOP_LEVEL_MODULE_NAME, MODULE_SEP_STRING, prefixes[i], n, NULL),
				    entity_domain);
    }

  return module;
}

entity module_name_to_entity(const string mn)
{
  /* Because of static C function, the entity returned is not always a
     top-level entity. */
  entity module = entity_undefined;

  if(static_module_name_p(mn)) {
    string cun = strdup(mn); /* compilation unit name */
    *(strstr(cun, FILE_SEP_STRING)+1)='\0';
    module = global_name_to_entity(cun, mn);
    free(cun);
  }
  else
    module = local_name_to_top_level_entity(mn);

  return module;
}

/* Retrieve an entity from its package/module name "m" and its local
   name "n". */
entity global_name_to_entity(string m, string n)
{
  return gen_find_tabulated(concatenate(m, MODULE_SEP_STRING, n, NULL),
			    entity_domain);
}

entity FindEntity(string package, string name)
{
  entity e = gen_find_tabulated(concatenate(package,MODULE_SEP_STRING,name,NULL), entity_domain);
  if( entity_undefined_p(e))
    e=gen_find_tabulated(concatenate(package,MODULE_SEP_STRING,"0",BLOCK_SEP_STRING,name,NULL), entity_domain);
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

entity find_or_create_entity(string full_name)
{
  entity e;

  if ((e = gen_find_tabulated(full_name, entity_domain))
      != entity_undefined) {
    pips_debug(8, "Entity \"%s\" is found\n", full_name);
  }
  else {
    pips_debug(8, "Entity \"%s\" is created\n", full_name);
    e = make_entity(strdup(full_name),
		    type_undefined, storage_undefined, value_undefined);
  }
  return e;
}



/* Problem: A functional global entity may be referenced without
   parenthesis or CALL keyword in a function or subroutine call.
   See SafeFindOrCreateEntity().
*/
entity FindOrCreateEntity(string package /* package name */,
			  string name /* entity name */)
{
  entity e = entity_undefined;

  e = find_or_create_entity(concatenate(package, MODULE_SEP_STRING, name, NULL));

  return e;
}


/* Return a top-level entity

   @param name of the entity to find/construct

   @return the entity
*/
entity FindOrCreateTopLevelEntity(string name)
{
  return FindOrCreateEntity(TOP_LEVEL_MODULE_NAME, name);
}


/* FIND_MODULE returns entity. Argument is module_name */
/* This function should be replaced by local_name_to_top_level_entity() */
/*entity find_entity_module(name)
string name;
{
    string full_name = concatenate(TOP_LEVEL_MODULE_NAME, 
				   MODULE_SEP_STRING, name, NULL);
    entity e = gen_find_tabulated(full_name, entity_domain);

    return(e);
}
*/

constant MakeConstantLitteral(void)
{
  return(make_constant(is_constant_litteral, NIL));
}

storage MakeStorageRom(void)
{
  return((make_storage(is_storage_rom, UU)));
}

/* END_EOLE */

value MakeValueUnknown(void)
{
  return(make_value(is_value_unknown, NIL));
}

/* returns a range expression containing e's i-th bounds */
expression entity_ith_bounds(entity e, int i)
{
  dimension d = entity_ith_dimension(e, i);
  syntax s = make_syntax(is_syntax_range,
			 make_range(copy_expression(dimension_lower(d)),
				    copy_expression(dimension_upper(d)),
				    make_expression_1()));
  return(make_expression(s, normalized_undefined));
}

/* true if e is an io instrinsic
 */
bool io_intrinsic_p(entity e)
{
  return top_level_entity_p(e) &&
    (ENTITY_WRITE_P(e) || ENTITY_REWIND_P(e) || ENTITY_OPEN_P(e) ||
     ENTITY_CLOSE_P(e) || ENTITY_READ_P(e) || ENTITY_BUFFERIN_P(e) ||
     ENTITY_BUFFEROUT_P(e) || ENTITY_ENDFILE_P(e) ||
     ENTITY_IMPLIEDDO_P(e) || ENTITY_FORMAT_P(e));
}

/* true if continue. See also macro ENTITY_CONTINUE_P
 */
bool entity_continue_p(entity f)
{
  return top_level_entity_p(f) &&
    same_string_p(entity_local_name(f), CONTINUE_FUNCTION_NAME);
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
	      pips_error("common_members_of_module",
			 "Varying size array \"%s\"\n", entity_name(v));
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

/* returns TRUE if l contains an entity with same type, local name and offset.
 */
static bool comparable_entity_in_list_p(entity common, entity v, list l)
{
  entity ref = entity_undefined;
  bool ok, sn, so = FALSE, st = FALSE;

  /* first find an entity with the same NAME.
   */
  string nv = entity_local_name(v);
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
  if (ok) ok = st = type_equal_p(entity_type(v), entity_type(ref));

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
  bool ok = TRUE;
  list /* of entity */ lv, lref;
  entity ref;
  pips_assert("entity is a common", entity_area_p(common));
  lv = area_layout(type_area(entity_type(common)));

  if (!lv) return TRUE; /* empty common! */

  /* take the first function as the reference for the check. */
  ref = ram_function(storage_ram(entity_storage(ENTITY(CAR(lv)))));
  lref = common_members_of_module(common, ref, FALSE);

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

  pips_internal_error("cannot find the %d dummy argument of %s\n",
		      rank, entity_name(the_fnct));

  return entity_undefined;
}

/* returns whether there is a main in the database
 */
extern gen_array_t db_get_module_list(void);

bool some_main_entity_p(void)
{
  gen_array_t modules = db_get_module_list();
  bool some_main = FALSE;
  GEN_ARRAY_MAP(name,
		if (entity_main_module_p(local_name_to_top_level_entity(name)))
		  some_main = TRUE,
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

/* 01/08/2003 Nga Nguyen -

   Since entity_local_name may contain PIPS special characters such as
   prefixes (label, common, struct, union, typedef, ...), this
   entity_user_name function is created to return the initial
   entity/variable name, as viewed by the user in his code.

   In addition, all possible seperators (file, module, block, member)
   are taken into account.

   Function strstr locates the occurence of the last special character
   which can appear just before the initial name, so the order of test
   is important.

   @return pointer to the the user name (not newly allocated!)
*/
string entity_user_name(entity e)
{
  string gn = entity_name(e);
  string un = global_name_to_user_name(gn);
  return un;
}

/* allocates a new string */
string entity_name_without_scope(entity e)
{
  string en = entity_name(e);
  string mn = entity_module_name(e);
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
string local_name_to_scope(string ln)
{
  string ns = strrchr(ln, BLOCK_SEP_CHAR);
  string s = string_undefined;
  extern string empty_scope(void);

  if(ns==NULL)
    s = empty_scope();
  else
    s = strndup(ln, ns-ln+1);

  pips_debug(8, "local name = \"%s\",  scope: \"%s\"\n",
	     ln, s);

  return s;
}

bool typedef_entity_p(entity e)
{
  /* Its name must contain the TYPEDEF_PREFIX just after the MODULE_SEP_STRING */
  string en = entity_name(e);
  string ms = strchr(en, MODULE_SEP);
  bool is_typedef = FALSE;

  if(ms!=NULL)
    is_typedef = (*(ms+1)==TYPEDEF_PREFIX_CHAR);

  return is_typedef;
}

bool member_entity_p(entity e)
{
  /* Its name must contain the MEMBER_PREFIX after the MODULE_SEP_STRING */
  string en = entity_name(e);
  string ms = strchr(en, MODULE_SEP);
  bool is_member = FALSE;

  if(ms!=NULL)
    is_member = (strchr(ms, MEMBER_SEP_CHAR)!=NULL);

  return is_member;
}

/* is p a formal parameter? */
bool entity_formal_p(entity p)
{
  storage es = entity_storage(p);
  bool is_formal = storage_formal_p(es);

  return is_formal;
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
entity MakeCompilationUnitEntity(string name)
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
     - The current module is a compilation unit and the entity is in the ram_shared list of 
     the ram storage of the compilation unit.
     - The current module is a normal function and the entity has a global scope.*/
  // Check if e belongs to module
  /* bool isbelong = TRUE;
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
    pips_internal_error("Unknown storage tag\n");

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
    string mn = entity_module_name(e);

    if(static_module_name_p(mn)) {
      m = gen_find_tabulated(concatenate(mn, MODULE_SEP_STRING, mn, NULL),entity_domain);
    }
    else {
      m = gen_find_tabulated(concatenate(TOP_LEVEL_MODULE_NAME,
					 MODULE_SEP_STRING, mn, NULL),
			     entity_domain);
    }
  }

  pips_assert("entity m is defined", !entity_undefined_p(m));

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
  functional ftf = type_functional(ft);
  bool mode_p = TRUE;

  /* Calls thru pointers require syntax_application */
  pips_assert("call to a function", type_functional_p(ft));

  if(!ENDP(functional_parameters(ftf))) {
    /* It is assumed that all parameters are passed the same way,
       either by valule or by reference */
    parameter p = PARAMETER(CAR(functional_parameters(ftf)));
    mode_p = (((int)mode_tag(parameter_mode(p)))==tag);
  }
  else {
    /* We are in trouble... because we have to call a higher-level
       function from the preprocessor library. */
    extern bool c_module_p(entity);
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
    sop = entity_intrinsic(PLUS_C_OPERATOR_NAME);
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
 * @return @a TRUE if e appears in one of the declaration in ldecl
 */
bool entity_used_in_declarations_p(entity e, list ldecl)
{
  bool found_p = FALSE;

  FOREACH(ENTITY, d, ldecl) {
    type dt = entity_type(d);
    list sel = type_supporting_entities(NIL, dt);

    if(gen_in_list_p(e, sel)) {
      pips_debug(8, "entity \"%s\" is used to declare entity \"%s\"\n",
		 entity_name(e), entity_name(d));
      found_p = TRUE;
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
 * @return @a TRUE if e appears in one of the declaration in ldecl
 */
bool type_used_in_type_declarations_p(entity e, list ldecl)
{
  bool found_p = FALSE;

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
	found_p = TRUE;
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
  free(variable_name);

  return ne;
}

/* Create a copy of an entity, with (almost) identical type, storage
   and initial value if move_initialization_p is FALSE, but with a slightly
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
entity make_entity_copy_with_new_name(entity e,
				      string global_new_name,
				      bool move_initialization_p)
{
  entity ne = entity_undefined;
  char * variable_name = strdup(global_new_name);
  int number = 0;

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

    /* FI: It would be better to perorm the memory allocation right
       away, instead of waiting for a later core dump in chains or
       ricedg, but I'm in a hurry. */
    ram_offset(r) = UNKNOWN_RAM_OFFSET;

    AddEntityToDeclarations(ne, m);
  }
  return ne;
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
    //pips_internal_error("Thread-safe entity \"%s\" redeclared\n", entity_name(v));
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
    //pips_internal_error("Thread-safe entity \"%s\" redeclared\n", entity_name(v));
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
  bool success_p = TRUE;

  FOREACH(ENTITY, e, el) {
    if(!check_entity(e)) {
      success_p = FALSE;
      break;
    }
  }
  return success_p;
}

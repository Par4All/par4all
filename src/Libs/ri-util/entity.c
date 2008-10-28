/* Functions closely related to the entity class, constructors, predicates,...
 *
 * $Id$
 */
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

static entity
make_empty_module(
    string full_name,
    type r)
{
    string name = string_undefined;
    entity e = gen_find_tabulated(full_name, entity_domain);
    entity DynamicArea, StaticArea;

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
		  make_code(NIL, strdup(""), make_sequence(NIL),NIL)));

    name = module_local_name(e);
    DynamicArea = FindOrCreateEntity(name, DYNAMIC_AREA_LOCAL_NAME);
    entity_type(DynamicArea) = make_type(is_type_area, make_area(0, NIL));
    entity_storage(DynamicArea) = MakeStorageRom();
    entity_initial(DynamicArea) = MakeValueUnknown();
    AddEntityToDeclarations(DynamicArea, e);

    StaticArea = FindOrCreateEntity(name, STATIC_AREA_LOCAL_NAME);
    entity_type(StaticArea) = make_type(is_type_area, make_area(0, NIL));
    entity_storage(StaticArea) = MakeStorageRom();
    entity_initial(StaticArea) = MakeValueUnknown();
    AddEntityToDeclarations(StaticArea, e);

    return(e);
}

entity 
make_empty_program(string name)
{
    string full_name = concatenate(TOP_LEVEL_MODULE_NAME, 
				   MODULE_SEP_STRING, MAIN_PREFIX, name, NULL);
    return make_empty_module(full_name, make_type(is_type_void, UU));
}

entity 
make_empty_subroutine(string name)
{
    string full_name = concatenate(TOP_LEVEL_MODULE_NAME, 
				   MODULE_SEP_STRING, name, NULL);
    return make_empty_module(full_name, make_type(is_type_void, UU));
}

entity 
make_empty_function(string name, type r)
{
    string full_name = concatenate(TOP_LEVEL_MODULE_NAME, 
				   MODULE_SEP_STRING, name, NULL);
    return make_empty_module(full_name, r);
}

entity
make_empty_blockdata(string name)
{
    string full_name = concatenate(TOP_LEVEL_MODULE_NAME, MODULE_SEP_STRING, 
				   BLOCKDATA_PREFIX, name, NULL);
    return make_empty_module(full_name, make_type(is_type_void, UU));
}


/* this function checks that e has an initial value code. if yes returns
it, otherwise aborts.  */

code 
EntityCode(e) 
entity e; 
{
    value ve = entity_initial(e);
    pips_assert("EntityCode", value_tag(ve) == is_value_code);
    return(value_code(ve));
}

entity 
make_label(string strg)
{

    entity l = make_entity(strdup(strg), type_undefined, storage_undefined, 
			value_undefined);
    entity_type(l) = (type) MakeTypeStatement();
    entity_storage(l) = (storage) MakeStorageRom();
    entity_initial(l) = make_value(is_value_constant, 
				   MakeConstantLitteral());
    return l;
}

/* This function returns a new label */
entity 
make_new_label(module_name)
char * module_name;
{
    /* FI: do labels have to be declared?*/
    /* FI: it's crazy; the name is usually derived from the entity
       by the caller and here the entity is retrieved from its name! */
    entity mod = local_name_to_top_level_entity(module_name); 
    string strg = new_label_name(mod);
    return make_label(strg);

}

entity 
make_loop_label(int __attribute__ ((unused)) desired_number,
		char * module_name) {
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
 */
string 
entity_local_name(entity e)
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

/* Returns a copy of the module local user name */
string module_local_name(entity e)
{
  /* No difference between modules and other entities, except for prefixes */
  /* Allocates a new string */

  string name = local_name(entity_name(e));

  return strdup(name 
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

string 
label_local_name(entity e)
{
    string name = local_name(entity_name(e));
    return name+strlen(LABEL_PREFIX);
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

/* In interprocedural context, returns the shortest non-ambiguous name
   for a variable. If it is local to the current module, use the user
   name. If not return entity_name(), which is not fully satisfying
   for C variables because it includes scope information.

   Note also that this function assumes the existence of a current module.
*/
string 
entity_minimal_name(entity e)
{
  entity m = get_current_module_entity();
  string local_name = module_local_name(m);

  pips_assert("some current entity", !entity_undefined_p(m));

  if (strcmp(module_local_name(m), entity_module_name(e)) == 0) {
    free(local_name);
    return global_name_to_user_name(entity_name(e));
  }
  else {
    free(local_name);
    return entity_name(e);
  }
}


string 
entity_and_common_name(entity e)
{
    entity m = get_current_module_entity();
    string name ;
    pips_assert("some current entity", !entity_undefined_p(m));
   
    name = concatenate(entity_local_name(ram_section(storage_ram(entity_storage(e)))),
		       MODULE_SEP_STRING,entity_name(e),NIL);
    
    return name +strlen(COMMON_PREFIX); 
}

bool 
entity_empty_label_p(entity e)
{
    return empty_label_p(entity_local_name(e));
}

bool 
entity_return_label_p(entity e)
{
    return return_label_p(entity_name(e));
}

bool 
entity_label_p(entity e)
{
    return type_statement_p(entity_type(e));
}

bool 
entity_module_p(entity e)
{
  if(typedef_entity_p(e))
    /* Functional typedef also have value code ... */
    return FALSE;
  else {
    value v = entity_initial(e);
    return v!=value_undefined && value_code_p(v);
  }
}

bool 
entity_main_module_p(entity e)
{
    return entity_module_p(e) &&
	strspn(entity_local_name(e), MAIN_PREFIX)==1;
}

bool 
entity_blockdata_p(entity e)
{
    return entity_module_p(e) && 
	strspn(entity_local_name(e), BLOCKDATA_PREFIX)==1;
}

bool 
entity_common_p(entity e)
{
    return entity_module_p(e) && /* ?????? */
	strspn(entity_local_name(e), COMMON_PREFIX)==1;
}

bool 
entity_function_p(entity e)
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

bool 
entity_subroutine_p(entity e)
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

bool entity_enum_p(entity e)
{
  return type_enum_p(entity_type(e));
}

bool entity_enum_member_p(entity e)
{
  value ev = entity_initial(e);

  pips_assert("Value of e is defined", !value_undefined_p(ev));
  return value_symbolic_p(ev);
}

bool 
local_entity_of_module_p(e, module)
entity e, module;
{
    bool
	result = same_string_p(entity_module_name(e), 
			       module_local_name(module));

    debug(6, "local_entity_of_module_p",
	  "%s %s %s\n", 
	  entity_name(e), result ? "in" : "not in", entity_name(module));

	  return(result);
}

bool 
entity_in_common_p(entity e)
{
    storage s = entity_storage(e);

    return(storage_ram_p(s) && 
	   !SPECIAL_COMMON_P(ram_section(storage_ram(s))));
}

string 
entity_module_name(entity e)
{
    return module_name(entity_name(e));
}

code 
entity_code(entity e)
{
    value ve = entity_initial(e);
    pips_assert("entity_code",value_code_p(ve));
    return(value_code(ve));
}

entity 
entity_empty_label(void)
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

bool 
top_level_entity_p(entity e)
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

bool 
io_entity_p(entity e)
{
    return(strncmp(IO_EFFECTS_PACKAGE_NAME, 
		   entity_name(e),
		   strlen(entity_module_name(e))) == 0);
}

bool 
intrinsic_entity_p(entity e)
{
  return (!value_undefined_p(entity_initial(e)) && value_intrinsic_p(entity_initial(e)));
}

/* FI: I do not understand this function name (see next one!). It seems to me
 * that any common or user function or user subroutine would
 * be returned.
 * FI: assert condition made stronger (18 December 1998)
 */
entity 
entity_intrinsic(string name)
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
   all be created beforehand by the bootstrap hase (see
   bootstrap/bootstrap.c). */

entity 
CreateIntrinsic(string name)
{
    entity e = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME, name);
    pips_assert("entity is defined", e!=entity_undefined && intrinsic_entity_p(e));
    return(e);
}

/* predicates on entities */

bool 
same_entity_p(entity e1, entity e2)
{
    return(e1 == e2);
}

/*  Comparison function for qsort.
 */
int 
compare_entities(entity *pe1, entity *pe2)
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
    return(strcmp(entity_name(*pe1), entity_name(*pe2)));
  }
}

/* sorted in place.
 */
void 
sort_list_of_entities(list l)
{
    gen_sort_list(l, compare_entities);
}

/*   TRUE if var1 <= var2
 */
bool 
lexicographic_order_p(entity var1, entity var2)
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
basic 
entity_basic(entity e)
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
bool 
entity_basic_p(entity e, int basictag)
{
    return (basic_tag(entity_basic(e)) == basictag);
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

entity 
local_name_to_top_level_entity(string n)
{
    entity module = entity_undefined;
    int i;

    /* Extension with C: the scope of a module can be its compilation unit if this is 
       a static module, not only TOP-LEVEL. */

    if (static_module_name_p(n)) {
      string cun = strdup(n);
      string sep = strchr(cun, FILE_SEP);
      string ln = strchr(n, FILE_SEP)+1;

      *(sep+1) = '\0';
      module = gen_find_tabulated(concatenate(cun, MODULE_SEP_STRING, ln, NULL),entity_domain);
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

entity module_name_to_entity(string mn)
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

entity 
global_name_to_entity(string m, string n)
{

    return gen_find_tabulated(concatenate(m, MODULE_SEP_STRING, n, NULL),
			      entity_domain);
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

entity 
find_or_create_entity(string full_name)
{
    entity e;
    
    if ((e = gen_find_tabulated(full_name, entity_domain)) 
	!= entity_undefined) {
	return e;
    }

    return make_entity(strdup(full_name),
		       type_undefined, storage_undefined, value_undefined);

}

/* Problem: A functional global entity may be referenced without
   parenthesis or CALL keyword in a function or subroutine call.
   See SafeFindOrCreateEntity().
*/

entity 
FindOrCreateEntity(
    string package, /* le nom du package */
    string name /* le nom de l'entite */)
{
    entity e = entity_undefined;

    e = find_or_create_entity(concatenate(package, MODULE_SEP_STRING, name, NULL));

    return e;
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

constant 
MakeConstantLitteral(void)
{
    return(make_constant(is_constant_litteral, NIL));
}

storage 
MakeStorageRom(void)
{
    return((make_storage(is_storage_rom, UU)));
}

/* END_EOLE */

value 
MakeValueUnknown(void)
{
    return(make_value(is_value_unknown, NIL));
}

/* returns a range expression containing e's i-th bounds */
expression 
entity_ith_bounds(entity e, int i)
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
bool
io_intrinsic_p(
    entity e)
{
    return top_level_entity_p(e) &&
        (ENTITY_WRITE_P(e) || ENTITY_REWIND_P(e) || ENTITY_OPEN_P(e) ||
	 ENTITY_CLOSE_P(e) || ENTITY_READ_P(e) || ENTITY_BUFFERIN_P(e) ||
	 ENTITY_BUFFEROUT_P(e) || ENTITY_ENDFILE_P(e) || 
	 ENTITY_IMPLIEDDO_P(e) || ENTITY_FORMAT_P(e));
}

/* true if continue
 */
bool 
entity_continue_p(
    entity f)
{
    return top_level_entity_p(f) && 
	same_string_p(entity_local_name(f), CONTINUE_FUNCTION_NAME);
}


/**************************************************** CHECK COMMON INCLUSION */

/* returns the list of entity to appear in the common declaration.
 */
list /* of entity */
common_members_of_module(
    entity common, 
    entity module,
    bool only_primary /* not the equivalenced... */)
{
  list result = NIL;
  int cumulated_offset = 0;
  pips_assert("entity is a common", type_area_p(entity_type(common)));

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

/* returns if l contains an entity with same type, local name and offset.
 */
static bool 
comparable_entity_in_list_p(
    entity common,
    entity v, 
    list l)
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
bool
check_common_inclusion(entity common)
{
    bool ok = TRUE;
    list /* of entity */ lv, lref;
    entity ref;
    pips_assert("entity is a common", type_area_p(entity_type(common)));
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
entity 
find_ith_formal_parameter(
    entity the_fnct,
    int    rank)
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
   Since entity_local_name may contain PIPS special characters 
   such as prefixes (label, common, struct, union, typedef, ...), 
   this entity_user_name function is created to return the 
   initial entity/variable name, as viewed by the user in his code.

   In addition, all possible seperators (file, module, block, member)
   are taken into account. 
   Function strstr locates the occurence of the last special character 
   which can appear just before the initial name, so the order of test is
   important.

   @return the user name in a new string (allocated with strdup!)
*/
string entity_user_name(entity e)
{
  string gn = entity_name(e);
  string un = strdup(global_name_to_user_name(gn));

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

  pips_debug(8, "entity name = \"%s\", without scope: \"%s\"\n",
	     en, enws);

  return enws;
}

/* allocates a new string */
string local_name_to_scope(string ln)
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


bool module_name_p(string name)
{
  return (!compilation_unit_p(name) && strstr(name, MODULE_SEP_STRING) == NULL);
}


bool static_module_name_p(string name)
{
  /* An entity is a static module if its name contains the FILE_SEP_STRING
     but the last one is not the last character of the name string */
  /* FI: I doubt this is true. Maybe if you're sure name is the name of a module? */
  return (!compilation_unit_p(name) && strstr(name, FILE_SEP_STRING) != NULL);
}

bool static_module_p(entity e)
{
  return static_module_name_p(entity_name(e));
}

bool compilation_unit_p(string module_name)
{
  /* A module name is a compilation unit if and only if its last character is
     FILE_SEP */
  if (module_name[strlen(module_name)-1]==FILE_SEP)
    return TRUE;
  return FALSE;
}

bool compilation_unit_entity_p(entity e)
{
  return compilation_unit_p(entity_name(e));
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
bool formal_entity_p(entity p)
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


entity MakeCompilationUnitEntity(string name)
{
  entity e = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME,name);

  pips_assert("name is a compilation unit name", compilation_unit_p(name));

  /* Normally, the storage must be rom but in order to store the list of entities
     declared with extern, we use the ram storage to put this list in ram_shared*/
  //entity_storage(e) = make_storage_ram(make_ram(entity_undefined,entity_undefined,0,NIL));
  entity_storage(e) = make_storage(is_storage_rom, UU);
  entity_type(e) = make_type_functional(make_functional(NIL,make_type_unknown()));
  entity_initial(e) = make_value(is_value_code, make_code(NIL,strdup(""), make_sequence(NIL),NIL));

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

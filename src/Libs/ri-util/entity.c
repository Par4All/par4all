#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "genC.h"
#include "misc.h"
#include "ri.h"

#include "ri-util.h"

static entity 
make_empty_module(
    string full_name,
    type r)
{
    string name = string_undefined;
    entity e = gen_find_tabulated(full_name, entity_domain);
    entity DynamicArea, StaticArea;

    /* FC: added to allow reintrance in HPFC 
     */
    if (e!=entity_undefined)
    {
	pips_user_warning("module %s already exists, returning it\n", 
			  full_name);
	return e;
    }

    pips_assert("undefined", e == entity_undefined);

    e = make_entity(strdup(full_name), 
		    make_type(is_type_functional, 
			      make_functional(NIL, r)),
		    MakeStorageRom(),
		    make_value(is_value_code,
			       make_code(NIL, strdup(""))));

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
    entity l;
  
    l = make_entity(strdup(strg), type_undefined, storage_undefined, 
			value_undefined);
    entity_type(l) = (type) MakeTypeStatement();
    entity_storage(l) = (storage) MakeStorageRom();
    entity_initial(l) = make_value(is_value_constant, MakeConstantLitteral());
 
    return(l);

}

entity 
make_loop_label(desired_number, module_name)
int desired_number;
char *module_name;
{
    entity e = make_new_label(module_name);
    return e;
}

/* predicates and functions for entities 
 */

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
    return e==NULL ? null_name : local_name(entity_name(e));
}

string 
module_local_name(entity e)
{
    string name = local_name(entity_name(e));
    return name 
	+ strspn(name, MAIN_PREFIX)
	+ strspn(name, BLOCKDATA_PREFIX)
	+ strspn(name, COMMON_PREFIX);
}

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

string 
entity_minimal_name(entity e)
{
    entity m = get_current_module_entity();

    pips_assert("some current entity", !entity_undefined_p(m));

    return (strcmp(module_local_name(m), entity_module_name(e)) == 0) ? 
	    entity_local_name(e) : entity_name(e) ;
}

bool 
entity_empty_label_p(entity e)
{
    return empty_label_p(entity_name(e));
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
    value v = entity_initial(e);

    return v!=value_undefined && value_code_p(v);
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
    return entity_module_p(e) && 
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
					   LABEL_PREFIX,
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
    value v = entity_initial(e);
    bool intrinsic_p = value_intrinsic_p(v);

    return intrinsic_p;
}

/* FI: I do not understand this function name. It seems to mee
 * that any common or user function or user subroutine would
 * be return.
 */
entity 
entity_intrinsic(string name)
{
    entity e = gen_find_tabulated(concatenate(TOP_LEVEL_MODULE_NAME,
					        MODULE_SEP_STRING,
					        name,
					        NULL),
				  entity_domain);

    pips_assert("entity_intrinsic", e != entity_undefined );
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
    else
	return(strcmp(entity_name(*pe1), entity_name(*pe2)));
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

    for(i=0; i<4 && entity_undefined_p(module); i++)
	module = gen_find_tabulated(concatenate
	  (TOP_LEVEL_MODULE_NAME, MODULE_SEP_STRING, prefixes[i], n, 0),
				    entity_domain);
    
    return module;
}

entity 
global_name_to_entity(string m, string n)
{

    return gen_find_tabulated(concatenate(m, MODULE_SEP_STRING, n, 0),
			      entity_domain);
}

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

entity 
FindOrCreateEntity(
    string package, /* le nom du package */
    string name /* le nom de l'entite */)
{
    return find_or_create_entity
	(concatenate(package, MODULE_SEP_STRING, name, 0));
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
                           make_range(gen_copy_tree(dimension_lower(d)),
                                      gen_copy_tree(dimension_upper(d)),
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


/* Functions closely related to the entity class, constructors, predicates,...
 *
 * $Id$
 *
 * $Log: entity.c,v $
 * Revision 1.53  2003/07/22 09:56:24  irigoin
 * function entity_global_name() added
 *
 * Revision 1.52  2002/06/27 14:44:07  irigoin
 * Function label_string_defined_in_statement_p() added
 *
 * Revision 1.51  2002/06/14 15:01:56  irigoin
 * Mostly new functions added to deal with labels and alternate return labels
 *
 * Revision 1.50  2002/06/13 12:06:02  irigoin
 * Adaptation to new ri.newgen with field "initializations" in structure "code"
 *
 * Revision 1.49  2002/05/02 15:32:54  coelho
 * module_local_name needed by eole.
 *
 * Revision 1.48  2002/03/08 10:12:35  irigoin
 * StackArea management adedd in common_members_of_module()
 *
 * Revision 1.47  2000/12/01 10:40:09  coelho
 * re debug...
 *
 * Revision 1.46  2000/12/01 10:35:03  coelho
 * suite du debug;-)
 *
 * Revision 1.45  2000/12/01 10:32:07  ancourt
 * debug string_to_entity_list
 *
 * Revision 1.44  2000/11/24 15:35:33  coelho
 * hop.
 *
 * Revision 1.43  1999/05/21 12:10:29  irigoin
 * Comment refined for CreateIntrinsic()
 *
 * Revision 1.42  1999/01/08 15:30:12  coelho
 * *** empty log message ***
 *
 * Revision 1.41  1999/01/08 14:20:29  coelho
 * some_main_entity_p function added.
 *
 * Revision 1.40  1998/12/18 19:53:59  irigoin
 * Stronger asserts put in CreateIntrinsic() and entity_intrinsic().
 *
 * Revision 1.39  1998/11/05 13:50:57  zory
 * new EOLE automatic inclusion tags added
 *
 * Revision 1.38  1998/11/05 09:28:45  zory
 * EOLE tags for automatic function extraction added
 *
 * Revision 1.37  1998/10/09 11:31:18  irigoin
 * common_members_of_module() updated because of the new heap area. RCS
 * fields added.
 *
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "linear.h"

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
			       make_code(NIL, strdup(""), make_sequence(NIL))));

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
make_loop_label(desired_number, module_name)
int desired_number;
char *module_name;
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

/* Used instead of the macro to pass as formal argument */
string entity_global_name(entity e)
{
    string null_name = "null";
    pips_assert("entity is defined", !entity_undefined_p(e));
    return entity_name(e);
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

string 
entity_minimal_name(entity e)
{
    entity m = get_current_module_entity();

    pips_assert("some current entity", !entity_undefined_p(m));

    return (strcmp(module_local_name(m), entity_module_name(e)) == 0) ? 
	    entity_local_name(e) : entity_name(e) ;
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

/* FI: I do not understand this function name (see next one!). It seems to mee
 * that any common or user function or user subroutine would
 * be return.
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

    e = find_or_create_entity(concatenate(package, MODULE_SEP_STRING, name, 0));

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

    MAP(ENTITY, v,
    {    
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
    },
        area_layout(type_area(entity_type(common))));

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

#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "misc.h"
#include "ri.h"

#include "ri-util.h"


bool variable_entity_p(e)
entity e;
{
    bool variable = entity_storage(e)!= storage_undefined && storage_ram_p(entity_storage(e));

    return variable;
}


/* this function adds an entity to the list of variables of the
CurrentFunction. it does nothing if e is already in the list. */

void AddEntityToDeclarations(e, f)
entity e;
entity f;
{
    cons *pc, *l;

    l = code_declarations(EntityCode(f));

    for (pc = l; pc != NULL; pc = CDR(pc)) {
	if (e == ENTITY(CAR(pc)))
		return;
    }

    code_declarations(EntityCode(f)) = CONS(ENTITY, e, l);
}


/*
void update_storage_of_variable_to_formal(v, m)
entity v;
entity m;
{
    entity_storage(fp) = make_storage(is_storage_formal, 
				      make_formal(m, 1));
}
*/


/*
 * entity make_scalar_entity(name, module_name, base)
 */
entity make_scalar_entity(name, module_name, base)
string name;
string module_name;
basic base;
{
    string 
	full_name;
    entity 
	e, f, a;
    basic 
	b = base;

    full_name =
	strdup(concatenate(module_name, MODULE_SEP_STRING, name, NULL));

    debug(8,"make_scalar_entity", "name %s\n", full_name);

    e = make_entity(full_name,
		    type_undefined, 
		    storage_undefined, 
		    value_undefined);

    entity_type(e) = (type) MakeTypeVariable(b, NIL);

    f = local_name_to_top_level_entity(module_name);

    a = global_name_to_entity(module_name, DYNAMIC_AREA_LOCAL_NAME); 

    entity_storage(e) = 
	make_storage(is_storage_ram,
		     make_ram(f, a,
			      (basic_tag(base)!=is_basic_overloaded)?
			      (add_variable_to_area(a, e)):(0),
			      NIL));

    entity_initial(e) = make_value(is_value_constant,
				   MakeConstantLitteral());

    return(e);
}


/*
 * looks for an entity which should be a scalar of the specified
 * basic. If found, returns it, else one is created.
 */
entity find_or_create_scalar_entity(name, module_name, base)
string name;
string module_name;
tag base;
{
    entity 
	e = entity_undefined;
    string 
	nom = concatenate(module_name, MODULE_SEP_STRING, name, NULL);

    if ((e = gen_find_tabulated(nom, entity_domain)) != entity_undefined) 
    {
	pips_assert("find_or_create_scalar_entity",
		    (entity_scalar_p(e) && 
		     entity_basic_p(e, base)));

	return(e);
    }

    return(make_scalar_entity(name, module_name, MakeBasic(base)));
}


entity make_scalar_integer_entity(name, module_name)
char *name;
char *module_name;
{
    string full_name;
    entity e, f, a ;
    basic b ;

    debug(8,"make_scalar_integer_entity", "begin name=%s, module_name=%s\n",
	  name, module_name);

    full_name = concatenate(module_name, MODULE_SEP_STRING, name, NULL);
    e = make_entity(strdup(full_name),
		    type_undefined, 
		    storage_undefined, 
		    value_undefined);

    b = make_basic(is_basic_int, 4); 

    entity_type(e) = (type) MakeTypeVariable(b, NIL);

    f = local_name_to_top_level_entity(module_name);
    a = global_name_to_entity(module_name, DYNAMIC_AREA_LOCAL_NAME);
    pips_assert("make_scalar_integer_entity", !entity_undefined_p(f) && !entity_undefined_p(a));

    entity_storage(e) = make_storage(is_storage_ram,
				     (make_ram(f, a,
					       add_variable_to_area(a, e),
					       NIL)));

    entity_initial(e) = make_value(is_value_constant,
				   MakeConstantLitteral());

    debug(8,"make_scalar_integer_entity", "end\n");

    return(e);
}


bool entity_scalar_p(e)
entity e;
{
    type t = entity_type(e);

    pips_assert("entity_scalar_p", type_variable_p(t));

    return(ENDP(variable_dimensions(type_variable(t))));
}


/* for variables (like I), not constants (like 1)!
 * use integer_constant_p() for constants
 */
bool entity_integer_scalar_p(e)
entity e;
{
    return(entity_scalar_p(e) &&
	   basic_int_p(variable_basic(type_variable(entity_type(e)))));
}


/* integer_scalar_entity_p() is obsolete; use entity_integer_scalar_p() */
bool integer_scalar_entity_p(e)
entity e;
{
    return type_variable_p(entity_type(e)) && 
	basic_int_p(variable_basic(type_variable(entity_type(e)))) &&
	    variable_dimensions(type_variable(entity_type(e))) == NIL;
}


dimension entity_ith_dimension(e, i)
entity e;
int i;
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
boolean entity_unbounded_p(e)
entity e;
{
    int nb_dim = NumberOfDimension(e);
    
    return(unbounded_dimension_p(entity_ith_dimension(e, nb_dim)));    
}



/* variable_entity_dimension(entity v): returns the dimension of variable v;
 * scalar have dimension 0
 */
int variable_entity_dimension(v)
entity v;
{
    int d = 0;

    pips_assert("variable_entity_dimension", type_variable_p(entity_type(v)));

    MAPL(cd, {
	d++;
    },
	 variable_dimensions(type_variable(entity_type(v))));

    return d;
}


void remove_variable_entity(v)
entity v;
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
 */
entity make_integer_constant_entity(c)
int c;
{
    entity ce;
    char *num = malloc(32);
    string cn;

    sprintf(num, "%d", c);
    cn = concatenate(TOP_LEVEL_MODULE_NAME,MODULE_SEP_STRING,num,(char *)NULL);
    ce = gen_find_tabulated(cn,entity_domain);
    if (ce==entity_undefined) {		/* make entity for the constant c */ 
	functional cf = 
	    make_functional(NIL, 
			    make_type(is_type_variable, 
				      make_variable(make_basic(is_basic_int,c),
						    NIL)));
	type ct = make_type(is_type_functional, cf);
	ce = make_entity(strdup(cn), ct, MakeStorageRom(),
			 make_value(is_value_constant, 
				    make_constant(is_constant_int, c)));
    }
    
    else 
	free(num);

    return(ce);
}


/* 
 * This function computes the current offset of the area a passed as
 * argument. The length of the variable v is also computed and then added
 * to a's offset. The initial offset is returned to the calling function.
 * v is added to a's layout.
 */
int add_variable_to_area(a, v)
entity a, v;
{
    int OldOffset=-1;
    type ta = entity_type(a);
    area aa = type_area(ta);

    if(top_level_entity_p(a)) {
	/* COMMONs are supposed to havethe same layout in each routine */
	pips_error("add_variable_to_area", "COMMONs should not be modified\n");
    }
    else {
	/* the local areas are StaticArea and DynamicArea */
	OldOffset = area_size(aa);
	area_size(aa) = OldOffset+SizeOfArray(v);
    }

    area_layout(aa) = gen_nconc(area_layout(aa), CONS(ENTITY, v, NIL));

    return(OldOffset);
}


void add_variable_declaration_to_module(m, v)
entity m;
entity v;
{
    value val = entity_initial(m);
    code c = code_undefined;

    pips_assert("add_variable_declaration_to_module", value_code_p(val));

    c = value_code(val);
    code_declarations(c) = gen_nconc(code_declarations(c),
				     CONS(ENTITY, v, NIL));
}


bool variable_in_module_p(v,m)
entity v;
entity m;
{
    bool in_module_1 = strcmp(module_local_name(m), entity_module_name(v)) == 0;
    bool in_module_2 = entity_is_argument_p(v, code_declarations (value_code(entity_initial (m))));

    pips_assert ("variable_in_module_p", in_module_1==in_module_2);

    return in_module_1;
}


/* Discard the decls_text string of the module code to make the
   prettyprinter ignoring the textual declaration and remake all from
   the declarations without touching the corresponding property
   (PRETTYPRINT_ALL_DECLARATIONS). RK, 31/05/1994. */
void discard_module_declaration_text(a_module)
entity a_module;
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
entity get_ith_dummy(prefix, suffix, i)
string prefix, suffix;
int i;
{
    char buffer[100]; 
    assert(i>=1 && i<=7);
    (void) sprintf(buffer, "%s%d", suffix, i);
    return(find_or_create_scalar_entity(buffer, prefix, is_basic_int));
}

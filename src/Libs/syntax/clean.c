#include <stdio.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"

#include "misc.h"
#include "ri-util.h"



static entity current_function = entity_undefined;
static char *current_function_local_name = NULL;


static void set_current_function(function)
entity function;
{
    pips_assert("set_current_function", top_level_entity_p(function));
    current_function = function;
    current_function_local_name = module_local_name(function);
}



static bool local_entity_of_current_function_p(e)
entity e;
{
    return(same_string_p(entity_module_name(e), current_function_local_name));
}


/* Useful when the user edit a source file and parse it again */
void CleanLocalEntities(function)
entity function;
{
    list function_local_entities;
    list pe;

    set_current_function(function);

    function_local_entities =
	gen_filter_tabulated(local_entity_of_current_function_p, 
			     entity_domain);

    /* FI: A memory leak is better than a dangling pointer? */
    /* My explanation: CleanLocalEntities is called by MakeCurrentFunction when
     * the begin_inst rule is reduced; by that time, the module entity and its
     * formal parameters have already been parsed and allocated... but they
     * are not yet fully initialized; thus the gen_full_free is a killer while
     * the undefinitions are ok.
     */
    /*gen_full_free_list(function_local_entities);*/

    for(pe=function_local_entities; !ENDP(pe); POP(pe)) {
	entity e = ENTITY(CAR(pe));
	storage s = entity_storage(e);

	pips_debug(8, "Clean up %s\n", entity_local_name(e));

	if(!storage_undefined_p(s) && storage_ram_p(s)) {
	    entity sec = ram_section(storage_ram(s));
	    type t = entity_type(sec);

	    pips_assert("CleanLocalEntities", type_area_p(t));

	    gen_remove(&(area_layout(type_area(t))),e);
	}

	/* for a FUNCTION XX, variable XX:XX has already been redefined
	 * when this piece of code is run; see previous comment (FI, 14/12/94)
	 */
	if(!(!storage_undefined_p(s) && storage_return_p(s))) {
	    entity_type(e) = type_undefined;
	    entity_storage(e) = storage_undefined;
	    entity_initial(e) = value_undefined;
	}

	if(!type_undefined_p(entity_type(e))) {
	  free_type(entity_type(e));
	  entity_type(e) = type_undefined;
	}
    }
}

/* Useful for ParserError()? */
void RemoveLocalEntities(function)
entity function;
{
    list function_local_entities;
    list pe;

    set_current_function(function);

    function_local_entities =
	gen_filter_tabulated(local_entity_of_current_function_p, 
			     entity_domain);

    /* FI: dangling pointers? Some variables may be referenced in area_layouts of
       global common entities! */
    /* gen_full_free_list(function_local_entities); */
    /* A gen_multi_recurse would be required but it's hard to be at the
       list level to remove the elements?!? */
    pips_assert("implemented", FALSE);
}

/* $RCSfile: call.c,v $ (version $Revision$)
 * $Date: 1996/06/18 15:57:17 $, 
 *
 * Fabien COELHO
 */

#include "local-header.h"

/* these functions must translate external reductions into local ones if 
 * must be. thus it is similar to summary effects call translations.
 * however, the functions there are heavily binded to effects...
 * while we would have prefered entities and references that are
 * shared between these two types (effect and reduction).
 * moreover I do not wish to chnage the code there too much. 
 * thus I first define a set of useful translation functions, and 
 * it will be up to both code to use them...
 * All these functions are clearly inspired by the ones there.
 * Fabien.
 */

static list /* of entity */
translate_into_local_entities(
    entity module,
    call c,
    entity var)
{
    storage st = entity_storage(var);

    pips_debug(7, "entity %s\n", entity_name(var));

    if (storage_formal_p(st))
    {
	int offset = formal_offset(storage_formal(st));
	syntax s = expression_syntax
	    (EXPRESSION(gen_nth(offset-1, call_arguments(c))));
	if (syntax_reference_p(s))
	    return CONS(ENTITY, reference_variable(syntax_reference(s)), NIL);
	/* else might be binded to some call */
    }
    
    return NIL;
}

static entity 
translate_into_one_local_entity(
    entity module, /* the module we're in */
    call c,        /* the called function */
    entity var)    /* the entity to be translated into the module space... */
{
    list le = translate_into_local_entities(module, c, var);
    if (gen_length(le)!=1)
    {
	gen_free_list(le);
	return entity_undefined;
    }
    else
    {
	entity e = ENTITY(CAR(le));
	gen_free_list(le);
	return e;
    }
}

/* translate formel_ref under called(real_args) into some new reference.
 * the formal reference may be arbitrary, A(B(I)) for instance...
 * thus reshapings must be managed with great care...
 * I'll do very basic things to begin with, translating only direct
 * one level references.
 */
bool /* whether ok */
translate_reference(
    call c,
    reference external_ref,
    reference *presult)
{
    entity var =
	translate_into_one_local_entity(get_current_module_entity(),
					c, reference_variable(external_ref));

    if (entity_undefined_p(var))
	return FALSE;

    *presult = make_reference(var, NIL);
    return TRUE;	
}

static bool /* whether ok */
translate_reduction(
    call c,
    reduction external_red,
    reduction *pred)
{
    reference ref;
    if (translate_reference(c, reduction_reference(external_red), &ref))
    {
	*pred = copy_reduction(external_red);
	free_reference(reduction_reference(*pred));
	reduction_reference(*pred) = ref;
	gen_free_list(reduction_dependences(*pred));
	reduction_dependences(*pred) = NIL;

	return TRUE;
    }

    return FALSE;
}

list /* of reduction */
translate_reductions(
    call c)
{
    entity fun = call_function(c);

    if (entity_module_p(fun))
    {
	reductions rs = load_summary_reductions(fun);
	list lr = NIL;
	reduction translated;

	MAP(REDUCTION, r,
	    if (translate_reduction(c, r, &translated))
	        lr = CONS(REDUCTION, translated, lr),
	    reductions_list(rs));
	
	return lr;
    }

    return NIL;
}
   

/* end of $RCSfile: call.c,v $
 */

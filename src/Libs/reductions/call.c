/* $RCSfile: call.c,v $ (version $Revision$)
 * $Date: 1996/06/21 11:47:52 $, 
 *
 * Fabien COELHO
 */

#include "local-header.h"

/* these functions must translate external reductions into local ones if 
 * must be. thus it is similar to summary effects call translations.
 * I do reuse some functions there by generating fake effects...
 * I think they should be cleaned so as to offer reference plus predicate
 * translations, and then would be used by reductions and effects.
 * Fabien.
 */

/* translation of a reference, based on effect translations...
 * such a interface should be available?
 */
list /* of reference */ 
summary_to_proper_reference(
    call c,
    reference r)
{
    effect e = make_effect(r, /* persistent! */
			   make_action(is_action_read, UU),
			   make_approximation(is_approximation_must, UU),
			   make_transformer(NIL, make_predicate(NULL)));
    list /* of effect */ le = CONS(EFFECT, e, NIL), lef, lref = NIL;
    
    lef = summary_to_proper_effects(call_function(c), call_arguments(c), le);
    
    MAP(EFFECT, ef, lref = CONS(REFERENCE, effect_reference(ef), lref), lef);
    
    gen_map(gen_free, lef); gen_free_list(lef);
    return lref;
}

static list /* of reduction */
translate_reduction(
    call c,
    reduction external_red)
{
    reference ref = reduction_reference(external_red);
    list /* of reference */ lref = summary_to_proper_reference(c, ref),
         /* of reduction */ lrds = NIL;

    MAP(REFERENCE, r, 
    {
	reduction red = copy_reduction(external_red);
	free_reference(reduction_reference(red));
	reduction_reference(red) = copy_reference(r); /* ??? */
	lrds = CONS(REDUCTION, red, lrds);
    },
	lref);

    gen_free_list(lref); /* just the backbone, refs are pointed to now */
    return lrds;
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

	MAP(REDUCTION, r, 
	    lr = gen_nconc(translate_reduction(c, r), lr),
	    reductions_list(rs));
	
	return lr;
    }

    return NIL;
}
   

/* end of $RCSfile: call.c,v $
 */

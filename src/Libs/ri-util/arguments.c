 /* package "arguments"
  *
  * Basic routines dealing with the arguments field of transformers
  * (i.e. list of entities, so it should be put in ri-util like many such
  * packages written for pips)
  *
  * Hash tables were not used because the argument lists are very short
  *
  * Francois Irigoin, April 1990
  */

#include <stdio.h>

#include "linear.h"

#include "genC.h"
#include "ri.h"
#include "misc.h"
#include "ri-util.h"
#include "constants.h"

void print_arguments(args)
cons * args;
{
    if(ENDP(args))
	(void) fprintf(stderr, "(nil)\n");
    else {
	MAPL(c, {entity e = ENTITY(CAR(c));
		 (void) fprintf(stderr,
			 c==args ? "%s" : ", %s",
			 e==entity_undefined? "entity_undefined" : entity_minimal_name(e));},
	     args);
	(void) putc('\n',stderr);
    }
}

void dump_arguments(args)
cons * args;
{
    if(ENDP(args))
	(void) fprintf(stderr, "(nil)\n");
    else {
	MAPL(c, {entity e = ENTITY(CAR(c));
		 (void) fprintf(stderr,
			 c==args ? "%s" : ", %s",
			 e==entity_undefined?
				"entity_undefined" : entity_name(e));},
	     args);
	(void) putc('\n',stderr);
    }
}

cons * arguments_add_entity(a, e)
cons * a;
entity e;
{
    if(!entity_is_argument_p(e, a))
	a = gen_nconc(a, CONS(ENTITY, e, NIL));
    return a;
}

cons * arguments_rm_entity(a, e)
cons * a;
entity e;
{
    if(entity_is_argument_p(e, a)) {
	gen_remove(&a, e);
    }
    else {
	pips_error("arguments_rm_entity", "entity %s is not in a\n",
		   entity_name(e));
    }

    return a;
}

/* cons * arguments_union(cons * a1, cons * a2): returns a = union(a1, a2)
 * where a1 and a2 are lists of entities.
 *
 * Entities in a1 have the same rank wrt a1 and a. Entities in a2 are likely
 * to have different ranks wrt a and a2. This might imply a transformer
 * renaming.
 */
cons * arguments_union(a1, a2)
cons * a1;
cons * a2;
{
    cons * a;

    if(a1==a2) {
	a = (cons *) gen_copy_seq(a1);
    }
    else {
	a = (cons *) gen_copy_seq(a1);
	MAPL(ce, {a = arguments_add_entity(a, ENTITY(CAR(ce)));}, a2);
    }

    return a;
}

bool arguments_equal_p(a1, a2)
cons * a1;
cons * a2;
{
    cons * ca1;
    cons * ca2;

    for( ca1 = a1, ca2 = a2; !ENDP(ca1) && !ENDP(ca2); POP(ca1), POP(ca2))
	if(ENTITY(CAR(ca1))!=ENTITY(CAR(ca2))) break;

    return ENDP(ca1) && ENDP(ca2);
}

bool entity_is_argument_p(e, args)
entity e;
cons * args;
{
    return gen_find_eq(e, args) != chunk_undefined;
}

cons * arguments_intersection(a1, a2)
cons * a1;
cons * a2;
{
    cons * a = NIL;
    MAPL(ca1, {
	entity e1 = ENTITY(CAR(ca1));
	if(entity_is_argument_p(e1, a2))
	    /* should gen_nconc be used ?!? Or is it only useful to
	     chain stuff at the end of a list? */
	    a = CONS(ENTITY, e1, a);
    },
	 a1);
    return a;
}

void free_arguments(args)
cons * args;
{
    /* should be a macro later, but keep debugging in mind! */
    gen_free_list(args);
}

cons * dup_arguments(args)
cons * args;
{
    /* should be a macro later, but keep debugging in mind! */
    return gen_copy_seq(args);
}

/* set difference: a1 - a2 ; similar to set intersection */
cons * arguments_difference(a1, a2)
cons * a1;
cons * a2;
{
    cons * a = NIL;
    MAPL(ca1, {
	entity e1 = ENTITY(CAR(ca1));
	if(!entity_is_argument_p(e1, a2))
	    /* should gen_nconc be used ?!? Or is it only useful to
	     chain stuff at the end of a list? */
	    a = CONS(ENTITY, e1, a);
    },
	 a1);
    return a;
}

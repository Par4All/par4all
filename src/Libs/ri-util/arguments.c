/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

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
/* Functions dealing with entity lists
 *
 * Called "arguments" because the package was developped within the
 * transformer library where entity lists were used to represent
 * transformer arguments. No specific link with transformers. Now used
 * here and there and moved into ri-util.
 */
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
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
#include "preprocessor.h"

void print_homogeneous_arguments(list args, const char* variable_name(entity))
{
    if(ENDP(args))
	(void) fprintf(stderr, "(nil)\n");
    else {
	MAPL(c, {entity e = ENTITY(CAR(c));
		 (void) fprintf(stderr,
			 c==args ? "%s" : ", %s",
			 e==entity_undefined? "entity_undefined" : variable_name(e));},
	     args);
	(void) putc('\n',stderr);
    }
}

void print_arguments(list args)
{
  print_homogeneous_arguments(args, entity_minimal_name);
}

/* entity_name is a macro, hence the code replication */
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
	pips_internal_error("entity %s is not in a",
		   entity_name(e));
    }

    return a;
}

/* cons * arguments_union(cons * a1, cons * a2): returns a = union(a1, a2)
 * where a1 and a2 are lists of entities. A new list is allocated.
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

/* Check the syntactic equality of lists a1 and a2 
 *
 * To check the equality of a1 and a2 as sets, use argument
 * intersection and a cardinal equality, assuming no entity occurs
 * more than once in a1 or a2.
 */
bool arguments_equal_p(list a1, list a2)
{
    list ca1;
    list ca2;

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

/* Build a new list with all entities occuring in both a1 and a2 */
list arguments_intersection(list a1, list a2)
{
    list a = NIL;
    FOREACH(ENTITY, e1, a1) {
	if(entity_is_argument_p(e1, a2))
	    /* should gen_nconc be used ?!? Or is it only useful to
	     chain stuff at the end of a list? */
	    a = CONS(ENTITY, e1, a);
    }

    return a;
}

/* Set equality of lists a1 and a2. Check that all entities in a1 also
 * occur in a2 and vice-versa.
 *
 * Might be faster to use the intersection and its cardinal...
 *
 * This algorithm is correct if an entity can appear several times in
 * a list.
 */
bool arguments_set_equal_p(list a1, list a2)
{
  bool set_equal_p = true;

  FOREACH(ENTITY, e1, a1) {
    if(!entity_is_argument_p(e1, a2)) {
      set_equal_p = false;
      break;
    }
  }
  if(set_equal_p) {
    FOREACH(ENTITY, e2, a2) {
      if(!entity_is_argument_p(e2, a1)) {
	set_equal_p = false;
	break;
      }
    }
  }

  return set_equal_p;
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

/* generate a Newgen list with all entities refered in vector b */
list base_to_entities(Pvecteur b)
{
  list el = NIL;
  Pvecteur ev;

  for(ev = b; ev!=NULL; ev = ev->succ) {
    entity e = (entity) vecteur_var(ev);
    el = CONS(ENTITY, e, el);
  }

  gen_nreverse(el);

  return el;
}

 /* NewGen interface with C3 type Psysteme for PIPS project 
  *
  * Systems of linear equalities and inequalities are stored as a triplet,
  * a base followed by two vector lists, the sets of equalities and 
  * inequalities: (b leq lineq). 
  *
  * Each vector list is also parenthesized:
  * (v1 v2 v3). Each vector itself (see Pvecteur.c) is a parenthesized
  * list of couples value/variable.
  *
  * For instance, the empty system (the whole space as defined by basis b) is:
  *    (b()())
  * where basis vector b is stored as a regular vector (see Pvecteur.c).
  *
  * To cope with the absence of unget() for f(), each vectors in the list
  * is separated by a space.
  *
  * Redundant information, the number of equalities and the number of 
  * inequalities and the system dimension, is discarded.
  *
  * Francois Irigoin, November 1990
  *
  * $Log: Psc.c,v $
  * Revision 1.10  2002/08/29 09:00:58  irigoin
  * sc_gen_copy_tree() now calls sc_copy() instead of sc_dup()
  *
  *
  *
  */

#include <stdio.h>
#include <string.h>

#include "linear.h"
#include "genC.h"
#include "ri.h"
#include "misc.h"

#include "newgen.h"

/* Sigh, sigh, sigh:
   - either ri-util.h must be included, as well as all underlaying libraries
   - or vect_gen_read() and vect_gen_write() must be locally declared,
   at the risk of a future inconsistency
*/

/* sigh no more, lady, sigh no more,
 * man, who decieves ever,
 * one foot in sea, and one on shore,
 * to one thing, constant never.
 * so sigh not so, 
 * but let them go,
 * and be your blith
 * ...
 * 
 * - I forgot some part of it I guess. FC
 */

#define error(fun, msg) { \
    fprintf(stderr, "pips internal error in %s: %s\n", fun, msg); exit(2); \
}

extern Pvecteur vect_gen_read();
extern void vect_gen_write();

void sc_gen_write(FILE *fd, Psysteme s)
{
    Pcontrainte c;
    Psysteme stored_s;
    static Psysteme undefined_s = SC_UNDEFINED;

    /* FI: we cannot not store SC_UNDEFINED as it is used in regions;
       we cannot store it like a system with an empty basis, no inequalities
       and no equalities because it is used to define transformer identity;
       conclusion: region library has to be changed and to use 
       transformer_undefined as context 

       Current kludge: SC_UNDEFINED is stored but retrieved as a system
       with 0 equalities and 0 inequalities over a space of dimension 0
       */
    if(SC_UNDEFINED_P(s)) {
	if(SC_UNDEFINED_P(undefined_s))
	    undefined_s = sc_make(CONTRAINTE_UNDEFINED, CONTRAINTE_UNDEFINED);
	stored_s = undefined_s;
    }
    else
	stored_s = s;

    pips_assert("sc_gen_write",!SC_UNDEFINED_P(stored_s));
    
    /*
      ifdebug(10){
      fprintf(stderr, "[sc_gen_write] sys 0x%x\n", (unsigned int) s);
      syst_debug(s); }
      */

    (void) fputc('(',fd);

    vect_gen_write(fd,stored_s->base);

    (void) fputc('(',fd);

    for (c = stored_s->egalites; c != NULL; c = c->succ) {
	(void) fputc(' ', fd);
	vect_gen_write(fd,c->vecteur);
    }

    (void) fputc(')',fd);

    (void) fputc('(',fd);

    for (c = stored_s->inegalites; c != NULL; c = c->succ) {
	(void) fputc(' ', fd);
	vect_gen_write(fd, c->vecteur);
    }

    (void) fputc(')',fd);

    (void) fputc(')',fd);

}

Psysteme sc_gen_read(FILE * fd /* ignored */, int (*f)())
{
    Psysteme s = sc_new();
    int c;

    if ((c = f()) != '(') {
	error("sc_gen_read","initial '(' missing\n");
    }

    s->base = vect_gen_read(fd, f);

    if ((c = f()) != '(') {
	error("sc_gen_read","equalities '(' missing\n");
    }

    while ((c = f()) != ')') {
	Pvecteur v = vect_gen_read(fd, f);
	Pcontrainte e= contrainte_make(v);

	pips_assert("sc_gen_read", c==' ');

	sc_add_egalite(s, e);
    }

    if ((c = f()) != '(') {
	error("sc_gen_read","inequalities '(' missing\n");
    }

    while ((c = f()) != ')') {
	Pvecteur v = vect_gen_read(fd, f);
	Pcontrainte i= contrainte_make(v);

	pips_assert("sc_gen_read", c==' ');

	sc_add_inegalite(s, i);
    }

    if ((c = f()) != ')') {
	error("sc_gen_read","closing ')' missing\n");
    }

    /* It might be a good idea to check that the basis is consistent
       with the equalities and inequalities that were read, later... */

    s->dimension = vect_size(s->base);

    /* FI: doesn't work because it's just the definition of 
       transformer_identity */
    /*
    if(s->dimension == 0) {
	pips_assert("sc_gen_read", s->nb_eq == 0 && s->nb_ineq == 0);
	sc_rm(s);
	s = SC_UNDEFINED;
    }
    */

    return s;
}

void sc_gen_free(Psysteme s)
{
    sc_rm(s);
}

Psysteme sc_gen_copy_tree(Psysteme s)
{
    return sc_copy(s);
}

int sc_gen_allocated_memory(Psysteme s)
{
    return contrainte_gen_allocated_memory(sc_egalites(s)) 
	 + contrainte_gen_allocated_memory(sc_inegalites(s)) 
	 + vect_gen_allocated_memory(sc_base(s)) 
	 + sizeof(Ssysteme) ;
}

/*   That is all
 */

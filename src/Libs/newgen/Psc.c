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
  */

#include <stdio.h>
#include <string.h>

#include "boolean.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

#include "genC.h"
#include "ri.h"
#include "ri-util.h"
#include "misc.h"

/* Sigh, sigh, sigh:
   - either ri-util.h must be included, as well as all underlaying libraries
   - or vect_gen_read() and vect_gen_write() must be locally declared,
   at the risk of a future inconsistency
   */

extern Pvecteur vect_gen_read();
extern void vect_gen_write();

void sc_gen_write(fd, s)
FILE *fd;
Psysteme s;
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

Psysteme sc_gen_read(fd, f)
FILE *fd; /* ignored */
int (*f)();
{
    Psysteme s = sc_new();
    int c;

    if ((c = f()) != '(') {
	pips_error("sc_gen_read","initial '(' missing\n");
    }

    s->base = vect_gen_read(fd, f);

    if ((c = f()) != '(') {
	pips_error("sc_gen_read","equalities '(' missing\n");
    }

    while ((c = f()) != ')') {
	Pvecteur v = vect_gen_read(fd, f);
	Pcontrainte e= contrainte_make(v);

	pips_assert("sc_gen_read", c==' ');

	sc_add_egalite(s, e);
    }

    if ((c = f()) != '(') {
	pips_error("sc_gen_read","inequalities '(' missing\n");
    }

    while ((c = f()) != ')') {
	Pvecteur v = vect_gen_read(fd, f);
	Pcontrainte i= contrainte_make(v);

	pips_assert("sc_gen_read", c==' ');

	sc_add_inegalite(s, i);
    }

    if ((c = f()) != ')') {
	pips_error("sc_gen_read","closing ')' missing\n");
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

void sc_gen_free(s)
Psysteme s;
{
    sc_rm(s);
    s = SC_UNDEFINED;
}

Psysteme sc_gen_copy_tree(s)
Psysteme s;
{
    return(sc_dup(s));
}

/* previously in /rice/debug.c */
void syst_debug(s)
Psysteme s;
{
    sc_fprint(stderr, s, entity_local_name);
}



void sc_sort(sc, sort_base)
Psysteme sc;
Pbase sort_base;
{
    sc_vect_sort(sc, compare_Pvecteur);
    sc->inegalites = 
	contrainte_sort(sc->inegalites, sc->base, sort_base, TRUE, TRUE);
    sc->egalites = 
	contrainte_sort(sc->egalites, sc->base, sort_base, TRUE, TRUE);
}



/*   That is all
 */




 /* package sc */

#include <stdio.h>
#include <string.h>
#include "arithmetique.h"
#include "boolean.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

/* Psysteme sc_inequations_elim_redund(Psysteme ps):
 *
 * Elimination des contraintes lineaires redondantes dans le systeme par
 * test de faisabilite. Le polyedre rationnel definit par ps peut etre
 * augmente par cette procedure qui utilise contrainte_reverse() et un
 * test de faisabilite: E(ps) peut etre strictement inclu dans E(ps'). Les
 * sommets rationnels du systeme generateur de ps ne sont pas respectes.
 *
 * Remarque: il ne faut pas appliquer de normalisation du systeme apres
 *  inversion de la contrainte et avant le test de faisabilite en RATIONELS, 
 * car l'elimination des redondances n'est alors pas  necessairement 
 * correcte en entiers.
 *
 *  resultat retourne par la fonction :
 *
 *  Psysteme	    : Le systeme initial est modifie. Il est egal a NULL si 
 *		      le systeme initial est non faisable.
 *
 *  Les parametres de la fonction :
 *
 *  Psysteme ps    : systeme lineaire 
 *
 */

Psysteme sc_inequations_elim_redund(ps)
Psysteme ps;
{
    Pcontrainte eq, eq1;

    for (eq = ps->inegalites;eq != NULL; eq = eq1) {
	eq1 = eq->succ;
	contrainte_reverse(eq);	
	if (sc_rational_feasibility_ofl_ctrl(ps, OFL_CTRL,TRUE))
	    contrainte_reverse(eq);
	else  { 
	    eq_set_vect_nul(eq);
	    sc_rm_empty_constraints(ps,0);
	}
    }
   
    return(ps);
}

/* Psysteme sc_elim_redund(Psysteme ps):
 * elimination des contraintes lineaires redondantes dans le systeme par test
 * de faisabilite. Les tests de faisabilite sont appliques sur tout le 
 * systeme. L'elimination des redondances est donc totale.
 *
 *  resultat retourne par la fonction :
 *
 *  Psysteme	    : Le systeme initial est modifie. Il est egal a NULL si 
 *		      le systeme initial est non faisable.
 *
 *  Les parametres de la fonction :
 *
 *  Psysteme ps    : systeme lineaire 
 *
 */

Psysteme sc_elim_redund(ps)
Psysteme ps;
{
    Pcontrainte eq;

    for (eq = ps->egalites; eq != NULL; eq=eq->succ)
	vect_normalize(eq->vecteur);
    ps = sc_kill_db_eg(ps);

    if (SC_UNDEFINED_P(ps)) return(ps);
    if (!sc_rational_feasibility_ofl_ctrl(ps,OFL_CTRL,TRUE))
    {
	sc_rm(ps);
	return(NULL);
    }
    
    ps = sc_inequations_elim_redund(ps);
    return(ps);
}

/* Same as above, but the basis is preserved and sc_empty is returned is
the system is not feasible. ps is assumed to be a consistent system of
constraints.  */
Psysteme sc_safe_elim_redund(ps)
Psysteme ps;
{
  Pbase b = base_copy(sc_base(ps));

  /* if (SC_UNDEFINED_P(ps)) return(ps); */

  ps = sc_elim_redund(ps);

  if(ps==NULL) {
    ps = sc_empty(b);
  }
  else {
    base_rm(b);
  }

  return(ps);
}

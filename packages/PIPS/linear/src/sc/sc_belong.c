/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

  This file is part of Linear/C3 Library.

  Linear/C3 Library is free software: you can redistribute it and/or modify it
  under the terms of the GNU Lesser General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  Linear/C3 Library is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public License
  along with Linear/C3 Library.  If not, see <http://www.gnu.org/licenses/>.

*/

 /* package sc */

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

/* Check if the integer point defined by vector v belongs to the set
   defined by the constraints of ps. The point is defined according to
   the basis b of ps. Any non-zero coefficient in pv is ignored if the
   corresponding variable does not appear in the basis of ps. An
   variable/dimension of ps that does not appear in pv has value 0. In
   other word, v is projected onto the space defined by b.

   This function is primarily defined to be called from the debugger
   gdb or in debugging code.
 */
bool sc_belongs_p(Psysteme ps, Pvecteur v)
{
  Pcontrainte eq;
  bool belongs_p = true;

  for (eq = ps->egalites;
       !CONTRAINTE_UNDEFINED_P(eq) && belongs_p;
       eq = eq->succ) {
    Pvecteur c = contrainte_vecteur(eq);
    belongs_p = equality_eval_p(c, v);
  }

  for (eq = ps->inegalites;
       !CONTRAINTE_UNDEFINED_P(eq) && belongs_p;
       eq = eq->succ) {
    Pvecteur c = contrainte_vecteur(eq);
    belongs_p = inequality_eval_p(c, v);
  }

  return belongs_p;
}

/* Check if the integer point defined by vector v is stricly inside
   the set defined by the constraints of ps. The point is defined
   according to the basis b of ps. Any non-zero coefficient in pv is
   ignored if the corresponding variable does not appear in the basis
   of ps. An variable/dimension of ps that does not appear in pv has
   value 0. In other word, v is projected onto the space defined by b.

   This function is defined to be called in the debugger gdb or in
   debugging code. But it may also be useful to check if a vertex of a
   generating system is significant or not with respect to ps. If it
   is found internal, it is not significant.
 */
bool sc_internal_p(Psysteme ps, Pvecteur v)
{
  Pcontrainte eq;
  bool internal_p = true;
  bool belongs_p = true;

  /* The equations must be met exactly */
  for (eq = ps->egalites;
       CONTRAINTE_UNDEFINED_P(eq) && belongs_p;
       eq = eq->succ) {
    Pvecteur c = contrainte_vecteur(eq);
    belongs_p = equality_eval_p(c, v);
  }

  /* For the convex set defined by ps to have some kind of inside, ps
     must contain at least one inequality. */
  if(CONTRAINTE_UNDEFINED_P(sc_inegalites(ps)))
    internal_p = false;

  for (eq = ps->inegalites;
       CONTRAINTE_UNDEFINED_P(eq) && belongs_p && internal_p;
       eq = eq->succ) {
    Pvecteur c = contrainte_vecteur(eq);
    Value r = contrainte_eval(c, v);
    internal_p = value_neg_p(r);
    // belongs_p = value_negz_p(r); seems redundant with internal_p, a
    // stronger condition
  }

  return internal_p;
}

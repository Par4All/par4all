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
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "database.h"

#include "ri-util.h"
#include "effects-util.h"
#include "constants.h"
#include "misc.h"
#include "text-util.h"
#include "text.h"



#include "effects-generic.h"
#include "effects-convex.h"

bool region_consistent_p(region reg)
{
#define MY_MAX_CHECK VALUE_CONST(100000000)
  bool consistent = true;
  Psysteme sc;
  Pbase b, t;
  Pcontrainte c;

  pips_assert("it is a region",  effect_consistent_p(reg));

  /* FI->BC: it might be better to use descriptors even for regions
     linked to store and type declarations. Not much time to think
     about it now. */
  if(!descriptor_none_p(effect_descriptor(reg))) {
    pips_assert("the descriptor is defined",  !descriptor_none_p(effect_descriptor(reg)));

    /* the system must be defined */
    consistent = consistent && !SC_UNDEFINED_P(region_system(reg));
    pips_assert("the region system is defined", consistent);

    /* check the Psysteme consistency */
    sc = region_system(reg);
    consistent = consistent && sc_weak_consistent_p(sc);
    pips_assert("the region system is consitent", consistent);

    /* the TCST variable must not belong to the base */
    b = sc_base(sc);
    for( t = b; !BASE_UNDEFINED_P(t) && consistent; t = t->succ)
      {
	consistent = consistent && !term_cst(t);
      }
    pips_assert("no TCST variable in the base", consistent);


    /* and there must be no high coefficient (or it should have raised an
     * exception before
     */
    c = sc_egalites(sc);
    while (c != (Pcontrainte) NULL)
      {
	Pvecteur v;
	for(v = c->vecteur; !VECTEUR_NUL_P(v); v = v->succ)
	  {
	    if (variable_phi_p((entity)vecteur_var(v)))
	      pips_assert("no high coefficient for PHI variables in region system.\n",
			  value_lt(vecteur_val(v),MY_MAX_CHECK));
	  }
	c = c->succ;
      }
    c = sc_inegalites(sc);
    while (c != (Pcontrainte) NULL)
      {
	Pvecteur v;
	for(v = c->vecteur; !VECTEUR_NUL_P(v); v = v->succ)
	  {
	    if (variable_phi_p((entity)vecteur_var(v)))
	      pips_assert("no high coefficient for PHI variables in region system.\n",
			  value_lt(vecteur_val(v),MY_MAX_CHECK));
	  }
	c = c->succ;
      }
  }
  return consistent;
}


bool regions_consistent_p(list l_reg)
{
    bool consistent = true;
    MAP(EFFECT, reg,
	consistent = consistent && region_consistent_p(reg),l_reg);
    return consistent;
}


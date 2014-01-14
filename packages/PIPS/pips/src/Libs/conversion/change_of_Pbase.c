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
/* package conversion
*/

#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>

#include "linear.h"

#include "genC.h"
#include "ri.h"
#include "ri-util.h"

#include "boolean.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"
#include "matrice.h"
#include "conversion.h"


void derive_new_basis(Pbase base_oldindex, Pbase * base_newindex, entity (*new_entity)(entity))
{
    Pbase pb;
    entity new_name;

    for (pb=base_oldindex; pb!=NULL; pb=pb->succ)
    {
	new_name = new_entity((entity) pb->var);
	base_add_dimension(base_newindex, (Variable) new_name);
    }
    *base_newindex = base_reversal(*base_newindex);
}

/* void change_of_base_index(Pbase base_oldindex, Pbase *base_newindex)
 * change of variable index from  base_oldindex to
 * base_newindex
*/
void change_of_base_index(base_oldindex, base_newindex)
Pbase base_oldindex;
Pbase *base_newindex;
{
    derive_new_basis(base_oldindex, base_newindex, make_index_prime_entity);
}

entity make_index_prime_entity(entity old_index)
{
  return make_new_index_entity(old_index, "p");
}

entity make_index_entity(entity old_index)
{
    return make_index_prime_entity(old_index);
}

/* Psysteme sc_change_baseindex(Psysteme sc, Pbase base_old, Pbase base_new)
 * le changement de base d'indice pour sc
 */
Psysteme sc_change_baseindex(sc, base_old, base_new)
Psysteme sc;
Pbase base_old;
Pbase base_new;
{
    Pbase pb1, pb2;

    for(pb1=base_old,pb2=base_new;pb1!=NULL;pb1=pb1->succ,pb2=pb2->succ)
	sc_variable_rename(sc,pb1->var,pb2->var);
    return(sc);
}

			    




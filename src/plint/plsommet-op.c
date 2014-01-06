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

 /* package plint */

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <stdio.h>
#include <stdlib.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

#include "sommet.h"

#define MALLOC(s,t,f) malloc(s)
#define FREE(s,t,f) free(s)

/* Psommet sommets_dupc(Psommet som):
 * copie d'une liste de sommets tout en respectant le meme ordre 
 */
Psommet sommets_dupc(som)
Psommet som;
{

    Psommet som1,som2;
    /* pointeur vers le dernier element de la liste*/
    Psommet pred;
    /* pointeur vers le premier element de la liste*/
    Psommet debut = NULL;


#ifdef TRACE
    printf(" ***** duplication du systeme \n");
#endif
    if (som)
    {
	som2 = sommet_dup(som);
	debut = som2;
	som2->eq_sat = (int *)MALLOC(sizeof(int),INTEGER,"sommets_dupc");
	*(som2->eq_sat) = *(som->eq_sat);
	pred = som2;
	for (som1 = som->succ; som1 != NULL; som1 = som1->succ)
	{

	    som2 = sommet_dup(som1);
	    pred->succ = som2;
	    som2->eq_sat =(int *)MALLOC(sizeof(int),INTEGER,
					"sommets_dupc");
	    *(som2->eq_sat) = *(som1->eq_sat);
	    pred = som2;
	}
    }
    return (debut);
}

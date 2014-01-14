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
 /* Code Generation for Distributed Memory Machines
  *
  * Generation of indices for the different bases used in code generation
  *
  *
  * PUMA, ESPRIT contract 2701
  *
  * Corinne Ancourt
  * 1991
  */

#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "genC.h"
#include "misc.h"

#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "dg.h"

typedef dg_arc_label arc_label;
typedef dg_vertex_label vertex_label;
#include "graph.h"

#include "matrice.h"
#include "tiling.h"
#include "ri-util.h"
#include "effects-util.h"
#include "wp65.h"

entity create_local_index(entity module, Pvecteur pv, string st)
{
    entity new_ind; 
    string name = strdup(concatenate(entity_local_name((entity) 
							vecteur_var(pv)),
				     st,NULL));
    if ((new_ind = 
	 gen_find_tabulated(concatenate(module_local_name(module), 
					MODULE_SEP_STRING,name,NULL),
				       entity_domain)) == entity_undefined)
    {
	new_ind=make_scalar_integer_entity(name, 
					    module_local_name(module));
    AddEntityToDeclarations( new_ind,module); 
    }
    free(name);
    return(new_ind);
}

entity create_local_index2(entity module, Pvecteur pv, string st)
{ 
    entity new_ind;
    string   name = strdup(concatenate(st,
			      entity_local_name((entity) vecteur_var(pv)),
			      NULL));
	if ((new_ind=gen_find_tabulated(concatenate
					(module_local_name(module), 
					 MODULE_SEP_STRING,name,NULL),
					entity_domain))== entity_undefined) {
	    new_ind=make_scalar_integer_entity(name, 
					       entity_local_name(
								 module));
	    AddEntityToDeclarations( new_ind,module);
	}
	free(name);
    return(new_ind);
}


void create_tile_basis(initial_module,compute_module,memory_module,initial_basis,tile_basis_in_initial_basis,
tile_basis_in_tile_basis,local_tile_basis,tile_basis_in_tile_basis2,local_tile_basis2)
entity initial_module;
entity compute_module;
entity memory_module;
Pbase initial_basis;
Pbase *tile_basis_in_initial_basis;
Pbase *tile_basis_in_tile_basis;
Pbase *local_tile_basis;
Pbase *tile_basis_in_tile_basis2;   /* idem que les precedentes mais pour la sousroutine compute_module*/
Pbase *local_tile_basis2;
{
    string name;
    entity new_ind;
    Pvecteur pv;
    Pbase tbib = BASE_NULLE;
    Pbase tbtb = BASE_NULLE;
    Pbase tbtbc = BASE_NULLE;
    Pbase ltb = BASE_NULLE;
    Pbase ltbc = BASE_NULLE;

    debug(8, "create_tile_basis", "begin initial_module_name=%s\n",
	  module_local_name(initial_module));
    ifdebug(8) {
	(void) fprintf(stderr, "initial_basis:\n");
	vect_fprint(stderr, initial_basis, (string(*)(void*))entity_local_name);
    }

    for (pv = initial_basis; !VECTEUR_NUL_P(pv); pv=pv->succ) {
	
	name = strdup(concatenate(entity_local_name((entity) 
						    vecteur_var(pv)),
				  SUFFIX_FOR_INDEX_TILE_IN_INITIAL_BASIS,
				  NULL));
	if ((new_ind = 
	     gen_find_tabulated(concatenate(module_local_name(initial_module), 
					    MODULE_SEP_STRING,name,NULL),
				entity_domain)) == entity_undefined)
	    new_ind=make_scalar_integer_entity(name, 
					       module_local_name(initial_module));
	
	free(name);
	tbib = vect_add_variable (tbib,(char *)new_ind);
	
	new_ind = create_local_index(compute_module,
				      pv,
				      SUFFIX_FOR_INDEX_TILE_IN_TILE_BASIS);
	tbtbc = vect_add_variable (tbtbc,(char *) new_ind);

	new_ind= create_local_index(memory_module,
				      pv,
				      SUFFIX_FOR_INDEX_TILE_IN_TILE_BASIS);
	tbtb = vect_add_variable (tbtb,(char *) new_ind);

	new_ind = create_local_index2(compute_module,  pv,
				     PREFIX_FOR_LOCAL_TILE_BASIS);
	ltbc = vect_add_variable (ltbc,(char *) new_ind);

	new_ind = create_local_index2(memory_module,  pv,
				     PREFIX_FOR_LOCAL_TILE_BASIS);
	ltb = vect_add_variable (ltb,(char *) new_ind);
    }


    *tile_basis_in_initial_basis = base_reversal(tbib);
    *tile_basis_in_tile_basis = base_reversal(tbtb);
    *local_tile_basis = base_reversal(ltb);
    *tile_basis_in_tile_basis2 = base_reversal(tbtbc);
    *local_tile_basis2 = base_reversal(ltbc);

    ifdebug(8) {
	(void) fprintf(stderr,"\nInitial basis:");
	base_fprint(stderr, initial_basis, (string(*)(void*))entity_local_name);
	(void) fprintf(stderr,"\nTile basis in initial basis:");
	base_fprint(stderr, *tile_basis_in_initial_basis, (string(*)(void*))entity_local_name);
	(void) fprintf(stderr,"\nTile basis in tile basis:");
	base_fprint(stderr, *tile_basis_in_tile_basis, (string(*)(void*))entity_local_name);
	(void) fprintf(stderr,"\nLocal basis:");
	base_fprint(stderr, *local_tile_basis, (string(*)(void*))entity_local_name);
	(void) fprintf(stderr,
		       "\nTile basis in tile basis for compute module:");
	base_fprint(stderr, *tile_basis_in_tile_basis2, (string(*)(void*))entity_local_name);
	(void) fprintf(stderr,"\nLocal basis for compute module:");
	base_fprint(stderr, *local_tile_basis2, (string(*)(void*))entity_local_name);
    }
    
    debug(8, "create_tile_basis", "end\n");
}


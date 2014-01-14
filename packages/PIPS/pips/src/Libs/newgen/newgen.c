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

#include "genC.h"

typedef void * void_star;
/* typedef void * vertex; */
typedef void * arc_label;
typedef void * vertex_label;
typedef void * db_void;
typedef void * operator_id_sons;

#include "linear.h"
#include "matrice.h"

#include "specs.h"
#include "all_newgen_headers.h"

#include "newgen.h"

void initialize_newgen()
{
    /* Read NewGen specification file
     */
    gen_read_spec(ALL_SPECS);
      
    /* Initialise external functions...
     * re-entry in newgen because of the graph stuff...
     */  
    gen_init_external(ARC_LABEL_NEWGEN_EXTERNAL, 
		      (void* (*)(FILE*,int(*)(void))) gen_read, 
		      (void (*)(FILE*, void*)) gen_write,
		      (void (*)(void*)) gen_free, 
		      (void* (*)(void*)) gen_copy_tree_with_sharing,
		      (int (*)(void*)) gen_allocated_memory);

    gen_init_external(VERTEX_LABEL_NEWGEN_EXTERNAL, 
		      (void* (*)(FILE*,int(*)(void))) gen_read, 
		      (void (*)(FILE*, void*)) gen_write,
		      (void (*)()) gen_free, 
		      (void* (*)()) gen_copy_tree_with_sharing,
		      (int (*)()) gen_allocated_memory);

    gen_init_external(PPOLYNOME_NEWGEN_EXTERNAL, 
		      (void* (*)()) polynome_gen_read, 
		      (void (*)()) polynome_gen_write, 
		      (void (*)()) polynome_gen_free,
		      (void* (*)()) polynome_gen_copy_tree,
		      (int (*)()) polynome_gen_allocated_memory);

    gen_init_external(PVECTEUR_NEWGEN_EXTERNAL, 
		      (void* (*)()) vect_gen_read, 
		      (void (*)()) vect_gen_write, 
		      (void (*)()) vect_gen_free,
		      (void* (*)()) vect_gen_copy_tree,
		      (int (*)()) vect_gen_allocated_memory);

    gen_init_external(PSYSTEME_NEWGEN_EXTERNAL, 
		      (void* (*)()) sc_gen_read, 
		      (void (*)()) sc_gen_write, 
		      (void (*)()) sc_gen_free,
		      (void* (*)()) sc_gen_copy_tree,
		      (int (*)()) sc_gen_allocated_memory);

    gen_init_external(MATRICE_NEWGEN_EXTERNAL, 
		      (void* (*)()) gen_core,
		      (void (*)()) gen_core, 
		      (void (*)()) free,
		      (void* (*)()) gen_core,
		      (int (*)()) NULL); /* can't get the size... FC */

    gen_init_external(PTSG_NEWGEN_EXTERNAL,
		      (void* (*)()) gen_core,
		      (void (*)()) gen_core, 
		      (void (*)()) sg_rm,
		      (void* (*)()) sg_dup,
		      (int (*)()) NULL); /* can't get the size... FC */

    gen_init_external(VOID_STAR_NEWGEN_EXTERNAL,
		      (void* (*)()) gen_core,
		      (void (*)()) gen_core, 
		      (void (*)()) gen_null,
		      (void* (*)()) gen_identity,
		      (int (*)()) gen_false);

    /* added because newgen lacks support for hash maps with integer keys */
    gen_init_external(OPERATOR_ID_SONS_NEWGEN_EXTERNAL,
		      (void* (*)()) gen_false, /* read */
		      (void (*)()) gen_null, /* write */
		      (void (*)()) gen_null, /* free */
		      (void* (*)()) gen_false, /* copy */
		      (int (*)()) gen_true); /* size */

    /* do nothing! */
    gen_init_external(DB_VOID_NEWGEN_EXTERNAL,
		      (void* (*)()) gen_false, /* read */
		      (void (*)()) gen_null, /* write */
		      (void (*)()) gen_null, /* free */
		      (void* (*)()) gen_false, /* copy */
		      (int (*)()) gen_true); /* size */
}

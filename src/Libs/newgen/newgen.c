/*
 * $Id$
 *
 * $Log: newgen.c,v $
 * Revision 1.20  2003/06/16 15:00:51  coelho
 * db_void added.
 *
 * Revision 1.19  1998/04/14 16:01:37  coelho
 * moved to an independent directory.
 *
 * Revision 1.18  1998/04/14 15:27:23  coelho
 * includes added.
 *
 * Revision 1.17  1998/04/11 13:01:09  coelho
 * includes needed newgen generated support functions.
 *
 */

#include <stdio.h>
#include <stdlib.h>

#include "genC.h"

typedef void * void_star;
/*typedef void * vertex;*/
typedef void * arc_label;
typedef void * vertex_label;
typedef void * db_void;

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
		      (char *(*)()) gen_read, 
		      (void (*)()) gen_write,
		      (void (*)()) gen_free, 
		      (char *(*)()) gen_copy_tree_with_sharing,
		      (int (*)()) gen_allocated_memory);

    gen_init_external(VERTEX_LABEL_NEWGEN_EXTERNAL, 
		      (char *(*)()) gen_read, 
		      (void (*)()) gen_write,
		      (void (*)()) gen_free, 
		      (char *(*)()) gen_copy_tree_with_sharing,
		      (int (*)()) gen_allocated_memory);

    gen_init_external(PPOLYNOME_NEWGEN_EXTERNAL, 
		      (char *(*)()) polynome_gen_read, 
		      (void (*)()) polynome_gen_write, 
		      (void (*)()) polynome_gen_free,
		      (char *(*)()) polynome_gen_copy_tree,
		      (int (*)()) polynome_gen_allocated_memory);

    gen_init_external(PVECTEUR_NEWGEN_EXTERNAL, 
		      (char *(*)()) vect_gen_read, 
		      (void (*)()) vect_gen_write, 
		      (void (*)()) vect_gen_free,
		      (char *(*)()) vect_gen_copy_tree,
		      (int (*)()) vect_gen_allocated_memory);

    gen_init_external(PSYSTEME_NEWGEN_EXTERNAL, 
		      (char *(*)()) sc_gen_read, 
		      (void (*)()) sc_gen_write, 
		      (void (*)()) sc_gen_free,
		      (char *(*)()) sc_gen_copy_tree,
		      (int (*)()) sc_gen_allocated_memory);

    gen_init_external(MATRICE_NEWGEN_EXTERNAL, 
		      (char *(*)()) gen_core,
		      (void (*)()) gen_core, 
		      (void (*)()) free,
		      (char *(*)()) gen_core,
		      (int (*)()) NULL); /* can't get the size... FC */

    gen_init_external(PTSG_NEWGEN_EXTERNAL,
		      (char *(*)()) gen_core,
		      (void (*)()) gen_core, 
		      (void (*)()) sg_rm,
		      (char *(*)()) sg_dup,
		      (int (*)()) NULL); /* can't get the size... FC */

    gen_init_external(VOID_STAR_NEWGEN_EXTERNAL,
		      (char *(*)()) gen_core,
		      (void (*)()) gen_core, 
		      (void (*)()) gen_null,
		      (char *(*)()) gen_identity,
		      (int (*)()) gen_false);

    /* do nothing! */
    gen_init_external(DB_VOID_NEWGEN_EXTERNAL,
		      (char* (*)()) gen_false, /* read */
		      (void (*)()) gen_null, /* write */
		      (void (*)()) gen_null, /* free */
		      (char* (*)()) gen_false, /* copy */
		      (int (*)()) gen_true); /* size */
}

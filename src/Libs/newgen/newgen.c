/* %A% ($Date: 1995/12/19 10:57:55 $, ) 
    version $Revision$, got on %D%, %T% [%P%].
   Copyright (c) - École des Mines de Paris Proprietary.  */

#ifndef lint
char top_level_newgen_c_vcid[] = "%A% ($Date: 1995/12/19 10:57:55 $, ) version $Revision$, got on %D%, %T% [%P%].\n Copyright (c) École des Mines de Paris Proprietary.";
#endif /* lint */
#include <stdio.h>

#include "genC.h"
#include "specs.h"

#include "ri.h"
#include "ri-util.h"
#include "complexity_ri.h"
#include "database.h"
#include "graph.h"
#include "dg.h"
#include "tiling.h"
#include "property.h"
#include "reduction.h"
#include "makefile.h"
#include "parser_private.h"
#include "hpf.h"
#include "hpf_private.h"
#include "message.h"
#include "paf_ri.h"
#include "word_attachment.h"

void initialize_newgen()
{
    /* Read NewGen specification file
     */
    gen_read_spec(ALL_SPECS);
      
    /* Initialise external functions...
     * re-entry in newgen because of the graph stuff...
     */  
    gen_init_external(ARC_LABEL, 
		      (char *(*)()) gen_read, 
		      (void (*)()) gen_write,
		      (void (*)()) gen_free, 
		      (char *(*)()) gen_copy_tree_with_sharing,
		      (int (*)()) gen_allocated_memory);

    gen_init_external(VERTEX_LABEL, 
		      (char *(*)()) gen_read, 
		      (void (*)()) gen_write,
		      (void (*)()) gen_free, 
		      (char *(*)()) gen_copy_tree_with_sharing,
		      (int (*)()) gen_allocated_memory);

    gen_init_external(PPOLYNOME, 
		      (char *(*)()) polynome_gen_read, 
		      (void (*)()) polynome_gen_write, 
		      (void (*)()) polynome_gen_free,
		      (char *(*)()) polynome_gen_copy_tree,
		      (int (*)()) polynome_gen_allocated_memory);

    gen_init_external(PVECTEUR, 
		      (char *(*)()) vect_gen_read, 
		      (void (*)()) vect_gen_write, 
		      (void (*)()) vect_gen_free,
		      (char *(*)()) vect_gen_copy_tree,
		      (int (*)()) vect_gen_allocated_memory);

    gen_init_external(PSYSTEME, 
		      (char *(*)()) sc_gen_read, 
		      (void (*)()) sc_gen_write, 
		      (void (*)()) sc_gen_free,
		      (char *(*)()) sc_gen_copy_tree,
		      (int (*)()) sc_gen_allocated_memory);

    gen_init_external(MATRICE, 
		      (char *(*)()) gen_core,
		      (void (*)()) gen_core, 
		      (void (*)()) free,
		      (char *(*)()) gen_core,
		      (int (*)()) NULL); /* can't get the size... FC */
}

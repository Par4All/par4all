/* 	%A% ($Date: 1995/06/28 11:03:57 $, ) version $Revision$, got on %D%, %T% [%P%].
        Copyright (c) - École des Mines de Paris Proprietary.	 */

#ifndef lint
static char vcid[] = "%A% ($Date: 1995/06/28 11:03:57 $, ) version $Revision$, got on %D%, %T% [%P%].\n Copyright (c) École des Mines de Paris Proprietary.";
#endif /* lint */
#include <stdio.h>

#include "genC.h"
#include "constants.h"

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
    /* lecture specifications NewGen */
    gen_read_spec(ALL_SPECS);
      
    /* initialisation des fonctions d'entrees-sorties pour les types
       de donnees non geres par NewGen */  
    gen_init_external(ARC_LABEL, 
		      (char *(*)()) gen_read, gen_write, gen_free, 
		      (char *(*)()) gen_copy_tree);
    gen_init_external(VERTEX_LABEL, 
		      (char *(*)()) gen_read, gen_write, gen_free,
		      (char *(*)()) gen_copy_tree);
    gen_init_external(PPOLYNOME, 
		      (char *(*)()) polynome_gen_read, polynome_gen_write, 
		      polynome_gen_free, (char *(*)()) polynome_gen_copy_tree);
    gen_init_external(PVECTEUR, 
		      (char *(*)()) vect_gen_read, vect_gen_write, 
		      vect_gen_free, (char *(*)()) vect_gen_copy_tree);
    gen_init_external(PSYSTEME, 
		      (char *(*)()) sc_gen_read, sc_gen_write, 
		      sc_gen_free, (char *(*)()) sc_gen_copy_tree);
    gen_init_external(MATRICE, 
		      (char *(*)()) matrice_gen_read, matrice_gen_write, 
		      matrice_gen_free, (char *(*)()) matrice_gen_copy_tree);
}

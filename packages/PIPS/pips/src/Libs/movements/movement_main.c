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
#include <string.h>
#include <setjmp.h>

#include "genC.h"       /* chunk is defined there, chunk was used by ri.h  */
#include "linear.h"
#include "ri.h"         /* used by ri-util.h */
#include "complexity_ri.h" /* useful, because PPOLYNOME is defined there */
#include "resources.h"  /* useful, because make is called directly by main */
#include "ri-util.h"    /* useful, because polynome_gen_write is called    */
#include "constants.h"  /* useful, because DBR_ is called directly by main */

jmp_buf pips_top_level;

main(argc, argv)
int argc;
char *argv[];
{
    entity mod;
    char *program_name;
    char *module_name;
    char *entities_filename;
    char *ppf_filename;
    FILE *fd;
    int i;
    bool prettyprint_it = false;
    statement s,stat;
    cons *copy_in = NIL;
    cons  *copy_out=NIL;
text t;
    

    /* get NewGen data type description */
    gen_read_spec(ALL_SPECS, (char*) NULL);
    gen_init_external(PVECTEUR, vect_gen_read, vect_gen_write, 
		      vect_gen_free, vect_gen_copy_tree);
    gen_init_external(PSYSTEME, sc_gen_read, sc_gen_write, 
		      sc_gen_free, sc_gen_copy_tree);
    gen_init_external(PPOLYNOME, polynome_gen_read, polynome_gen_write, 
		      polynome_gen_free, polynome_gen_copy_tree);


    module_name = argv[2];

    fprintf(stderr, "----------------------in main-----------------------\n");

    db_open_workspace(argv[1]);
    if(setjmp(pips_top_level)) {
	/* case you come back from a user_error */
	db_close_workspace();
	exit(1);
    }
    else {
	db_open_module(module_name);

	make(DBR_CODE, module_name);

	s = (statement) db_get_memory_resource(DBR_CODE, module_name, false);

	mod = local_name_to_top_level_entity(module_name); 
	search_array_from_statement2(module_name,s,&copy_in,&copy_out);

	
db_close_module(module_name);
	user_log("I have done this: ...\n");
    }
    db_close_workspace();
    fprintf(stderr, "----------------------out of main-----------------------\n");
}

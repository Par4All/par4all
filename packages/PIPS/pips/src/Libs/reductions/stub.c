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
/*
  
  Stub.c -- 

  Interface to CommonLISP's implementation of Generalized Reductions 

  P. Jouvelot

*/


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "ri-util.h"
#include "effects-util.h"
#include "misc.h"
#include "database.h"
#include "resources.h"

#include "pipsdbm.h"

#include "control.h"

#define LISP (getenv("LISP"))

#define REDUCTIONS_SOURCE_DIRECTORY \
        (strdup(concatenate(getenv("DEVEDIR"), "/Lib/reductions", NULL)))

bool old_reductions(string mod_name)
{
    string program_name = db_get_current_workspace_name() ;
    char *command ;
    struct stat buf ;
    char *options ; 
    statement mod_stat ;

    if( stat( LISP, &buf ) != 0 ) {
	user_warning("reductions",
		     "No lisp! Skipping reduction detection\n") ;
	/* true or FALSE? After all, it's just a user warning, not an error */
	return false;
    }
    debug_on("REDUCTIONS_DEBUG_LEVEL");
    db_close_workspace(false) ;
    options = (get_debug_level() <= 5) ? " -batch" : "" ;
    asprintf(&command, 
	    "(echo \"(defparameter files-directory \\\"%s\\\")\
             (load (concatenate %s files-directory \\\"/init\\\"))\
             (load (concatenate %s files-directory \\\"/top\\\"))\
             (reductions t \\\"%s\\\" \\\"%s\\\")\") | %s%s", 
            REDUCTIONS_SOURCE_DIRECTORY,
            "'string",
            "'string",
	    program_name, 
	    mod_name, 
	    LISP,
	    options ) ;
    debug( 5, "reductions", "\ncommand = %s\n", command ) ;

    if( (system( command ) >> 8) != 0 ) {
	pips_internal_error("Lisp process died unexpectedly" ) ;
    }
    free(command);
    debug_off();
    db_open_workspace( program_name ) ;
    mod_stat = (statement) 
	    db_get_memory_resource(DBR_CODE, mod_name, true);
    module_reorder( mod_stat ) ;
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, strdup(mod_name), mod_stat);

    return true;
}

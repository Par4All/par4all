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
#include "ri-util.h"
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
    char command[ 1024 ] ;
    struct stat buf ;
    char *options ; 
    statement mod_stat ;

    if( stat( LISP, &buf ) != 0 ) {
	user_warning("reductions",
		     "No lisp! Skipping reduction detection\n") ;
	/* TRUE or FALSE? After all, it's just a user warning, not an error */
	return FALSE;
    }
    debug_on("REDUCTIONS_DEBUG_LEVEL");
    db_close_workspace(FALSE) ;
    options = (get_debug_level() <= 5) ? " -batch" : "" ;
    sprintf(command, 
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
	pips_error( "reductions", "Lisp process died unexpectedly\n" ) ;
    }
    debug_off();
    db_open_workspace( program_name ) ;
    mod_stat = (statement) 
	    db_get_memory_resource(DBR_CODE, mod_name, TRUE);
    module_reorder( mod_stat ) ;
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, strdup(mod_name), mod_stat);

    return TRUE;
}

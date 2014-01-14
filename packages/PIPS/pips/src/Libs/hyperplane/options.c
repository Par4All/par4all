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
 /* package hyperplane
  *
  * Yi-qing Yang, June 1990
  */

#include <stdio.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "constants.h"

#include "misc.h"

#include "arithmetique.h"
#include "boolean.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

#include "properties.h"
#include "hyperplane.h"

/* analyzes command line options and returns the index of the first argument 
 * in argv[]
 */
int set_hyperplane_parameters(argc, argv)
int argc;
char * argv[];
{
    int errflg = 0;
    char * options = HYPERPLANE_OPTIONS;
    bool truth = true;
    char c;
    extern char * optarg;
    extern int optind;

    /* get Pips default properties */
    (void) pips_flag_p(PRETTYPRINT_TRANSFORMER);

    /* set semantics default properties */
    set_bool_property(PRETTYPRINT_EXECUTION_CONTEXT, true);
    set_bool_property(SEMANTICS_FLOW_SENSITIVE, true);

    /* check and set execution parameters; should be done with
       pips_flag_get() from the parameters package to avoid inconsistencies */
    while((c=getopt(argc,argv,options))!=-1) {
	switch(c) {
	case '-':
	    /* let's take care of that later */
	    ;
	case 'O':
		set_bool_property(SEMANTICS_STDOUT, truth);
	    break;
	case 't':
		set_bool_property(PRETTYPRINT_TRANSFORMER, truth);
	    break;
	case 'c':
		set_bool_property(PRETTYPRINT_EXECUTION_CONTEXT, truth);
	    break;
	case 'i':
		set_bool_property(SEMANTICS_INTERPROCEDURAL, truth);
	    break;
	case 'f':
		set_bool_property(SEMANTICS_FLOW_SENSITIVE, truth);
	    break;
	case 'e':
		set_bool_property(SEMANTICS_INEQUALITY_INVARIANT, truth);
	    if(truth) {
		/* implied flags, non resettable */
		set_bool_property(SEMANTICS_FIX_POINT, true);
		set_bool_property(SEMANTICS_FLOW_SENSITIVE, true);
	    }
	    break;
	case 'o':
		/* set_bool_property(ONE_TRIP_DO, truth); */
	    break;
	case 'd':
		set_bool_property(SEMANTICS_FIX_POINT, truth);
	    if(truth) 
		/* this implies flow sensitivity, i.e. convex hull 
		   computation; non resettable */
		set_bool_property(SEMANTICS_FLOW_SENSITIVE, truth);
	    break;
	case 'D':
	    if(!truth) 
		/* debug level cannot be inversed, but can be zeroed */
		set_int_property(SEMANTICS_DEBUG_LEVEL, 0);
	    else {
		int semantics_debug_level = atoi(optarg);
	    /* a 0 debug level is equivalent to no debug 
	       a negative debug level is meaningless */
	    if(semantics_debug_level<=0) errflg++;
		else set_int_property(SEMANTICS_DEBUG_LEVEL,
				      semantics_debug_level);
	    }
	    break;
	case '?':
	default:
	    errflg++;
	}
	/* set/reset truth */
	truth = (c != '-');
	}

    /* two parameters at least are needed */
    if (argc - optind < 2 || errflg != 0)
	user_error("semantics_main",
		   "usage: %s [-tcife[D nn]] program module [module ...]\n", 
		   argv[0]);


    /*
      (void) fprintf(stderr,"semantics_main semantics_debug_level = %d\n",
      semantics_debug_level);
      (void) fprintf(stderr,"semantics_main argc = %d, optind=%d \n",
      argc, optind);
      */

    return optind;
}

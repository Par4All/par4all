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
/* Find out a legal hyperplane direction. In a first phase, trust the user blindly!
 */

#include <stdio.h>
#include <strings.h>

#include "boolean.h"
#include "arithmetique.h"
#include "matrice.h"
#include "genC.h"
#include "misc.h"

bool
interactive_hyperplane_direction(Value * h, int n)
{
    int i;
    int n_read;
    string resp = string_undefined;
    string cn = string_undefined;
    bool return_status = false;

    /* Query the user for h's coordinates */
    pips_assert("hyperplane_direction", n>=1);
    debug(8, "interactive_hyperplane_direction", "Reading h\n");
    resp = user_request("Hyperplane direction vector?\n"
			"(give all its integer coordinates on one line): ");
    if (resp[0] == '\0') {
	user_log("Hyperplane loop transformation has been cancelled.\n");
	return_status = false;
    }
    else {    
	cn = strtok(resp, " \t");

	return_status = true;
	for( i = 0; i<n; i++) {
	    if(cn==NULL) {
		user_log("Not enough coordinates. "
			 "Hyperplane loop transformation has been cancelled.\n");
		return_status = false;
		break;
	    }
	    n_read = sscanf(cn," " VALUE_FMT, h+i);
	    if(n_read!=1) {
		user_log("Not enough coordinates. "
			 "Hyperplane loop transformation has been cancelled.\n");
		return_status = false;
		break;
	    }
	    cn = strtok(NULL, " \t");
	}
    }

    if(cn!=NULL) {
	user_log("Too many coordinates. "
		 "Hyperplane loop transformation has been cancelled.\n");
	return_status = false;
    }

    ifdebug(8) {
	if(return_status) {
	    pips_debug(8, "Hyperplane direction vector:\n");
	    for( i = 0; i<n; i++) {
		(void) fprintf(stderr," " VALUE_FMT, *(h+i));
	    }
	    (void) fprintf(stderr,"\n");
	    pips_debug(8, "End\n");
	}
	else {
	    pips_debug(8, "Ends with failure\n");
	}
    }

    return return_status;
}

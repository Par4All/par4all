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
  * Reads and prints the target machine description (compatible formats)
  * Temporary version. A more general machine model should be defined by
  * Lei for complexity evaluation.
  *
  * File: model.c
  *
  * PUMA, ESPRIT contract 2701
  *
  * Francois Irigoin
  * 1991
  */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "genC.h"
#include "misc.h"
#include "top-level.h" /* ??? */
#include "properties.h"

void model_fprint(FILE * fd, int pn, int bn, int ls)
{
    fprintf(fd, 
	    "Target Machine:\n"
	    "Processor number: %d\n"
	    "Memory bank number: %d\n"
	    "Bank width (i.e. line size): %d\n\n", pn, bn, ls);
}

void model_fscan(FILE * fd, int * ppn, int * pbn, int * pls)
{
    int i1, i2, i3, i4;
    i1 = fscanf(fd, "Target Machine:\n");
    i2 = fscanf(fd, "Processor number: %d\n", ppn);
    i3 = fscanf(fd, "Memory bank number: %d\n", pbn);
    i4 = fscanf(fd, "Bank width (i.e. line size): %d\n\n", pls);
    if(i1!=0 && i2!=1 && i3!=1 && i4!=1) {
	user_error("model_fscan", "Bad format for machine model\n");
    }
}

void get_model(int * ppn, int * pbn, int * pls)
{
    FILE * fd;
    const char* model_rc = get_string_property("WP65_MODEL_FILE");
    if ((fd = fopen(model_rc, "r")) == NULL) {
	fd = fopen_config(model_rc, NULL,NULL);
    }

    model_fscan(fd, ppn, pbn, pls);
}

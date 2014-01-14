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

#include "newgen_version.h"
#include "linear_version.h"
#include "pips_version.h"

/* could be shared somewhere? */
char * soft_revisions = 
  "  newgen:  " NEWGEN_REV "\n"
  "  linear:  " LINEAR_REV "\n"
  "  pips:    " PIPS_REV "\n"
  "  nlpmake: " PIPS_MAKE_REV "\n";

char * soft_date = UTC_DATE;

char * cc_version = CC_VERSION;

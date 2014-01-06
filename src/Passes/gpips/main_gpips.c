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
 * This file contains the main for gpips.
 * Please, do not change anything! do any change to wpips_main().
 *
 * FC.
 */
/*
 * forked to main_gpips.c
 * Edited by Johan GALL
 *
 */

extern char * pips_thanks(char *, char *);
extern int gpips_main(int, char**);

int main(int argc, char ** argv)
{
    pips_thanks("gpips", argv[0]);
    return gpips_main(argc, argv);
}

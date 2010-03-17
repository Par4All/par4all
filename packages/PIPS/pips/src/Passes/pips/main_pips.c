/*

  $Id$

  Copyright 1989-2010 MINES ParisTech

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
/* 
 * This file contains the main for pips.
 * Please, do not change anything! do any change to pips_main().
 */

extern char * pips_thanks(char *, char *);
extern int pips_main(int, char**);

int main(int argc, char ** argv)
{
    pips_thanks("pips", argv[0]);
    return pips_main(argc, argv);
}

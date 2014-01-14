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
#define Entier int
#define FORMAT "%d"
#define UN 1

/* Modification by Alexis Platonoff: we need a greater solution size */
/*#define SOL_SIZE 4096*/
#define SOL_SIZE 16384

#define True 1
#define False 0

#ifdef TC
#define DEBUG 8
#endif

#define Q if(cross_product>=limit)

#define MAXCOL 1024
#define MAXPARM 100


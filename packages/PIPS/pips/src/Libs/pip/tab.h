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
struct A
    {struct A *precedent;
     char *bout;
    };

struct L
    {int flags;
     union { int unit;
             Entier * val;
	   } objet;
    };

#define Unit 1
#define Plus 2
#define Minus 4
#define Zero 8
#define Critic 16
#define Unknown 32

#define Sign 62

#define Index(p,i,j) (p)->row[i].objet.val[j]
#define Flag(p,i)    (p)->row[i].flags
struct T
    {int height, width;
     struct L row[1];
    };

typedef struct T Tableau;

void tab_init();

char * tab_hwm();

void tab_reset();

Tableau * tab_alloc();

void tab_copy();

Tableau * tab_get();

void tab_display();

Tableau *tab_expand();

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

#ifndef lint
char vcid_pip_tab[] = "$Id$";
#endif /* lint */

#include <stdio.h>
#include <stdlib.h>

/* Newgen includes 	*/
#include "genC.h"

/* C3 includes 		*/
#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "sc.h"
#include "polyedre.h"
#include "matrix.h"

/* Pips includes 	*/
#include "ri.h"
#include "constants.h"
#include "ri-util.h"
#include "misc.h"
#include "bootstrap.h"
#include "complexity_ri.h"
#include "database.h"
#include "graph.h"
#include "dg.h"
#include "paf_ri.h"
#include "parser_private.h"
#include "property.h"
#include "reduction.h"
#include "text.h"
#include "paf-util.h"
#include "pip.h"


#define max(x,y) (x>y? x : y)
#define TAB_CHUNK 4096*sizeof(Entier)

static char *tab_free, *tab_top;
static struct A *tab_base;
extern int allocation;
extern long int cross_product, limit;

void tab_init()
{
 tab_free = (char*) malloc(sizeof (struct A));
 if(!tab_free)
     {printf("tab_init : la machine est trop petite\n");
      exit(21);
     }
 allocation = 1;
 tab_top = tab_free + sizeof (struct A);
 tab_base = (struct A *)tab_free;
 tab_free += sizeof(struct A);
 tab_base->precedent = NULL;
 tab_base->bout = tab_top;
}

char * tab_hwm()
{
 return(tab_free);
}

void tab_reset(p)
char *p;
{struct A *g;
 while(tab_base)
     {
      if((char *)tab_base + sizeof(struct A) <= p && p <= tab_top) break;
      g = tab_base->precedent;
      free(tab_base);
      allocation--;
      tab_base = g;
      tab_top = tab_base->bout;
     }
 if(tab_base) tab_free = p;
 else printf("tab_reset : panique\n");
}

Tableau * tab_alloc(h, w, n)
int h, w, n;
/* h : le nombre de ligne reelles;
   n : le nombre de lignes virtuelles
*/
{
 char *p; Tableau *tp;
 Entier *q;
 long taille;
 int i;
 taille = 2 * sizeof(int) + (h+n) * sizeof (struct L)
          + h * w * sizeof (Entier);
 if(tab_free + taille >= tab_top)
     {struct A * g;
      long int d;
      d = max(TAB_CHUNK, taille + sizeof(struct A));
      tab_free = (char*) malloc(d);
      if(!tab_free)
          {printf("tab_alloc : plus de place\n");
           exit(23);
          }
      allocation++;
      g = (struct A *)tab_free;
      g->precedent = tab_base;
      tab_top = tab_free + d;
      tab_free += sizeof(struct A);
      tab_base = g;
      g->bout = tab_top;
     }
 p = tab_free;
 tab_free += taille;
 tp = (Tableau *)p;
 q = (Entier *)(p +  2 * sizeof(int) + (h+n) * sizeof (struct L));
 for(i = 0; i<n ; i++)
     {tp->row[i].flags = Unit | Zero;
      tp->row[i].objet.unit = i;
     }
 for(i = n; i < (h+n); i++)
     {tp->row[i].flags = 0;
      tp->row[i].objet.val = q;
      q += w;
     }
 tp->height = h + n; tp->width = w;
 return(tp);
}

Tableau * tab_get(foo, h, w, n)
FILE * foo;
int h, w, n;
{
 Tableau *p;
 int i, j, c;
 p = tab_alloc(h, w, n);
 while((c = dgetc(foo)) != EOF)if(c == '(')break;
 for(i = n; i<h+n; i++)
     {p->row[i].flags = Unknown;
      while((c = dgetc(foo)) != EOF)if(c == '[')break;
      for(j = 0; j<w; j++)
	if(dscanf(foo, FORMAT, p->row[i].objet.val+j) < 0)
	        return NULL;
      while((c = dgetc(foo)) != EOF)if(c == ']')break;
     }
 while((c = dgetc(foo)) != EOF)if(c == ')')break;
 return((Tableau *) p);
}

char *Attr[] = {"Unit", "+", "-", "0", "*", "?"};

void tab_display(p, foo)
FILE *foo;
Tableau *p;
{

 int i, j, ff, fff, n;
 Entier x;
 fprintf(foo, "%ld/[%d * %d]\n", cross_product, p->height, p->width);
 for(i = 0; i<p->height; i++)
     {fff = ff = p->row[i].flags;
      if(ff == 0) break;
      n = 0;
      while(fff)
          {if(fff & 1) fprintf(foo, "%s ",Attr[n]);
	   n++; fff >>= 1;
	  }
      fprintf(foo, "#[");
      if(ff & Unit)
          for(j = 0; j<p->width; j++)
	      fprintf(foo, " /%d/",(j == p->row[i].objet.unit)? 1: 0);
      else
          for(j = 0; j<p->width; j++)
	      {x = Index(p,i,j);
	       fprintf(foo, FORMAT, x);
	       fprintf(foo, " ");
	      }
      fprintf(foo, " ]\n");
     }
}


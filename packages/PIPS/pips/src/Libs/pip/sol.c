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
#include "pip__type.h"

#include <stdio.h>

extern long int cross_product, limit;
extern int verbose;
extern FILE *dump;

struct S
    {int flags;
     Entier param1, param2;
    };

#define Nil  1
#define If   2
#define List 3
#define Form 4
#define New  5
#define Div  6
#define Val  7

struct S sol_space[SOL_SIZE];
static int sol_free;

Entier sol_mod();

Entier sol_pgcd(x,y)
Entier x, y;
{Entier r;
 while(y)
     {r = sol_mod(x, y);
      x = y;
      y = r;
     }
 return(x>= 0? x : -x);
}

Entier sol_ppcm(Entier x, Entier y)
{
    Entier gcd = sol_pgcd(x, y), lcm = x * (y/gcd);
    return lcm>=0 ? lcm: -lcm;
}

void sol_init()
{
 sol_free = 0;
}

int sol_hwm()
{
 return(sol_free);
}

void sol_reset(p)
int p;
{
 if(p<0 || p>=SOL_SIZE)
     {fprintf(stderr, "salades de saison .... : sol\n");
      exit(40);
     }
 sol_free = p;
}

struct S *sol_alloc()
{struct S *r;
 r = sol_space + sol_free;
 sol_free++;
 if(sol_free >= SOL_SIZE)
     {fprintf(stderr, "grosse solution!!! : sol\n");
      exit(26);
     }
 return(r);
}

void sol_nil()
{
 struct S * r;
 r = sol_alloc();
 r -> flags = Nil;
 if(verbose > 0)
   {fprintf(dump, "sol_nil\n");
    fflush(dump);
  }
}

int is_not_Nil(p)
int p;
{
 return(sol_space[p].flags != Nil);
}

void sol_if()
{
 struct S *r;
 r = sol_alloc();
 r -> flags = If;
}

void sol_list(n)
int n;
{struct S * r;
 r = sol_alloc();
 r->flags = List;
 r->param1 = n;
}

void sol_forme(l)
int l;
{
 struct S *r;
 r = sol_alloc();
 r -> flags = Form;
 r -> param1 = l;
}

void sol_new(k)
int k;
{
 struct S *r;
 r = sol_alloc();
 r -> flags = New;
 r -> param1 = k;
}

void sol_div()
{
 struct S *r;
 r = sol_alloc();
 r -> flags = Div;
}

void sol_val(n, d)
Entier n, d;
{
 struct S *r;
 r = sol_alloc();
 r -> flags = Val;
 r -> param1 = n;
 r -> param2 = d;
}

int sol_edit(foo, i)
FILE *foo;
int i;
{int j, n;
 struct S *p;
 Entier N, D, d;
 debut : p = sol_space + i;
 switch(p->flags)
    {case Nil : fprintf(foo, "()\n");
                if(verbose>0)fprintf(dump, "()\n");
                i++; break;
     case If  : fprintf(foo, "(if ");
                if(verbose>0)fprintf(dump, "(if ");
                i = sol_edit(foo, ++i);
		i = sol_edit(foo, i);
		i = sol_edit(foo, i);
		fprintf(foo, ")\n");
                if(verbose>0)fprintf(dump, ")\n");
		break;
     case List: fprintf(foo, "(list ");
                if(verbose>0)fprintf(dump, "(list ");
                n = p->param1;
                i++;
                while(n--) i = sol_edit(foo, i);
		fprintf(foo, ")\n");
                if(verbose>0)fprintf(dump, ")\n");
		break;
     case Form: fprintf(foo, "#[");
                if(verbose>0)fprintf(dump, "#[");
                n = p->param1;
                for(j = 0; j<n; j++)
                   {i++; p++;
                    N = p->param1; D = p->param2;
                    d = sol_pgcd(N, D);
	            if(d == D){putc(' ', foo);
		               fprintf(foo, FORMAT, N/d);
			       if(verbose>0){
				 putc(' ', dump);
				 fprintf(dump, FORMAT, N/d);
			       }
			      }
                    else{putc(' ', foo);
		         fprintf(foo,FORMAT,N/d);
		         fprintf(foo,"/");
			 fprintf(foo,FORMAT, D/d);
			 if(verbose>0)
			   {putc(' ', dump);
			    fprintf(dump,FORMAT,N/d);
			    fprintf(dump,"/");
			    fprintf(dump,FORMAT, D/d);
			  }
			}
                   }
		fprintf(foo, "]\n");
	        if(verbose>0)fprintf(dump, "]\n");
		i++;
		break;
     case New : n = p->param1;
                fprintf(foo, "(newparm %d ", n);
		if(verbose>0)fprintf(dump, "(newparm %d ", n);
                i = sol_edit(foo, ++i);
		fprintf(foo, ")\n");
		if(verbose>0)fprintf(dump, ")\n");
		goto debut;
     case Div : fprintf(foo, "(div ");
	        if(verbose>0)fprintf(dump, "(div ");
                i = sol_edit(foo, ++i);
		i = sol_edit(foo, i);
		fprintf(foo, ")\n");
		if(verbose>0)fprintf(dump, ")\n");
		break;
     case Val : N = p->param1; D = p->param2;
                d = sol_pgcd(N, D);
		if(d == D){putc(' ', foo);
		           fprintf(foo, FORMAT, N/d);
			   if(verbose>0)
			     {putc(' ', dump);
			      fprintf(dump, FORMAT, N/d);
			    }
			  }
		else{putc(' ', foo);
		     fprintf(foo, FORMAT, N/d);
		     fprintf(foo, "/");
		     fprintf(foo, FORMAT, D/d);
		     if(verbose>0)
		       {putc(' ', dump);
			fprintf(dump, FORMAT, N/d);
			fprintf(dump, "/");
			fprintf(dump, FORMAT, D/d);
		      }
		    }
		i++;
		break;
     default  : fprintf(foo, "Inconnu : sol\n");
		if(verbose>0)fprintf(dump, "Inconnu : sol\n");
    }
    return(i);
}

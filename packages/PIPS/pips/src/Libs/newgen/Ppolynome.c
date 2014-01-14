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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "linear.h"

#include "genC.h"
#include "ri.h"
#include "misc.h"

#include "newgen.h"

void monome_gen_write(FILE *fd, Pmonome pm)
{
    pips_assert("monome_gen_write", !MONOME_UNDEFINED_P(pm));

    fprintf(fd, "{%f< ", monome_coeff(pm));
    vect_gen_write(fd, monome_term(pm));
    fprintf(fd, ">}");
}

Pmonome monome_gen_read(fd, f)
FILE *fd;
int (*f)();
{
    Pmonome pm = (Pmonome) malloc(sizeof(Smonome));
    char buffer[128];
    int c, ibuffer = 0;

    if ( (c = f()) != '{' ) {
	pips_internal_error("initial '{' missing!");
    }

    while ( (c = f()) != '<' ) { 
	if ( ibuffer >= 127 )
	    pips_internal_error("vecteur '<' missing!");
	buffer[ibuffer++] = c;
    }
    buffer[ibuffer] = '\0';

    sscanf(buffer, "%f", &monome_coeff(pm));

    if ( (c = f()) == ' ' ) {
	monome_term(pm) = (Pvecteur)vect_gen_read(fd, f);
    }

    if ( (c = f()) != '>' ) {
	pips_internal_error("closing '>' missing!");
    }

    if ( (c = f()) != '}' ) {
	pips_internal_error("closing '}' missing!");
    }

    return pm;
}

void monome_gen_free(pm)
Pmonome pm;
{
    monome_rm(&pm);
}

Pmonome monome_gen_copy_tree(pm)
Pmonome pm;
{
    return(monome_dup(pm));
}

void polynome_gen_write(fd,pp)
FILE *fd;
Ppolynome pp;
{
    Ppolynome p;

    pips_assert("polynome_gen_write", !POLYNOME_UNDEFINED_P(pp));

    fprintf(fd, "\n[");

    for(p=pp; !POLYNOME_NUL_P(p); p=polynome_succ(p)) {
	Pmonome pm = polynome_monome(p);

	fprintf(fd, " ");
	monome_gen_write(fd, pm);
    }

    fprintf(fd, "]\n");
}

Ppolynome polynome_gen_read(fd, f)
FILE *fd;
int (*f)();
{
    Ppolynome pp = POLYNOME_NUL;
    int c;

    if ( (c = f()) != '\n' ) {
	pips_internal_error("initial newline missing!");
    }

    if ( (c = f()) != '[' ) {
	pips_internal_error("initial '[' missing!");
    }

    while ( (c = f()) != ']' ) {
	Pmonome pm = monome_gen_read(fd, f);
	Ppolynome p = monome_to_new_polynome(pm);

	pips_assert("polynome_gen_read", c==' ');

	polynome_add(&pp, p);
    }

    if ( (c = f()) != '\n' ) {
	pips_internal_error("closing newline missing!");
    }

    return pp;
}

void polynome_gen_free(pp)
Ppolynome pp;
{
    polynome_rm(&pp);
}

Ppolynome polynome_gen_copy_tree(pp)
Ppolynome pp;
{
    return(polynome_dup(pp));
}

int 
monome_gen_allocated_memory(
    Pmonome m)
{
    return sizeof(Smonome) + vect_gen_allocated_memory(m->term);
}

int
polynome_gen_allocated_memory(
    Ppolynome p)
{
    int result = 0;
    for(; p; p=p->succ)
	result += monome_gen_allocated_memory(p->monome) + sizeof(Spolynome);
    return result;
}

/* Ppolynome.c
 * NewGen interface with type Ppolynome for PIPS project
 *
 * Lei Zhou       Feb. 27, 91
 *
 * Modifications:
 *     
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "linear.h"

#include "genC.h"
#include "ri.h"
#include "ri-util.h"
#include "misc.h"

#include "boolean.h"
#include "vecteur.h"
#include "polynome.h"

/*#include "ri-util.h"*/

void monome_gen_write(fd,pm)
FILE *fd;
Pmonome pm;
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
	pips_error("monome_gen_read","initial '{' missing!\n");
    }

    while ( (c = f()) != '<' ) { 
	if ( ibuffer >= 127 )
	    pips_error("monome_gen_read","vecteur '<' missing!\n");
	buffer[ibuffer++] = c;
    }
    buffer[ibuffer] = '\0';

    sscanf(buffer, "%f", &monome_coeff(pm));

    if ( (c = f()) == ' ' ) {
	monome_term(pm) = (Pvecteur)vect_gen_read(fd, f);
    }

    if ( (c = f()) != '>' ) {
	pips_error("monome_gen_read","closing '>' missing!\n");
    }

    if ( (c = f()) != '}' ) {
	pips_error("monome_gen_read","closing '}' missing!\n");
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
	pips_error("polynome_gen_read","initial newline missing!\n");
    }

    if ( (c = f()) != '[' ) {
	pips_error("polynome_gen_read","initial '[' missing!\n");
    }

    while ( (c = f()) != ']' ) {
	Pmonome pm = monome_gen_read(fd, f);
	Ppolynome p = monome_to_new_polynome(pm);

	pips_assert("polynome_gen_read", c==' ');

	polynome_add(&pp, p);
    }

    if ( (c = f()) != '\n' ) {
	pips_error("polynome_gen_read","closing newline missing!\n");
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

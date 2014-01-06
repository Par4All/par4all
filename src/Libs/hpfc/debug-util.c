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
/* this is a set of functions to help hpfc debugging
 *
 * Fabien Coelho, May 1993.
 */

#include "defines-local.h"
#include "pipsdbm.h"



void print_align(align a)
{
    (void) fprintf(stderr, "aligned (%zd dimensions)\n", 
		   gen_length(align_alignment(a)));
    gen_map((gen_iter_func_t)print_alignment, align_alignment(a));
    (void) fprintf(stderr, "to template %s\n\n",
		   entity_name(align_template(a)));
}

void print_alignment(alignment a)
{
    (void) fprintf(stderr,
		   "Alignment: arraydim %"PRIdPTR", templatedim %"PRIdPTR",\n",
		   alignment_arraydim(a),
		   alignment_templatedim(a));
    
    (void) fprintf(stderr,"rate: ");
    print_expression(alignment_rate(a));
    (void) fprintf(stderr,"\nconstant: ");
    print_expression(alignment_constant(a));
    (void) fprintf(stderr,"\n");
}

void print_aligns(void)
{
    fprintf(stderr,"Aligns:\n");
    MAP(ENTITY, a,
     {
	 (void) fprintf(stderr, "of array %s\n", entity_name(a));
	 print_align(load_hpf_alignment(a));
	 (void) fprintf(stderr,"\n");
     },
	list_of_distributed_arrays());
}

void print_distributes(void)
{
    fprintf(stderr,"Distributes:\n");

    MAP(ENTITY, t,
     {
	 (void) fprintf(stderr, "of template %s\n", entity_name(t));
	 print_distribute(load_hpf_distribution(t));
	 (void) fprintf(stderr,"\n");
     },
	 list_of_templates());
    
}

void print_distribute(distribute d)
{
    (void) fprintf(stderr,"distributed\n");

    gen_map((gen_iter_func_t)print_distribution, distribute_distribution(d));

    (void) fprintf(stderr, "to processors %s\n\n", 
		   entity_name(distribute_processors(d)));    
}

void print_distribution(distribution d)
{
    switch(style_tag(distribution_style(d)))
    {
    case is_style_none:
	(void) fprintf(stderr,"none, ");
	break;
    case is_style_block:
	(void) fprintf(stderr,"BLOCK(");
	print_expression(distribution_parameter(d));
	(void) fprintf(stderr,"), ");
	break;
    case is_style_cyclic:
	(void) fprintf(stderr,"CYCLIC(");
	print_expression(distribution_parameter(d));
	(void) fprintf(stderr,"), ");
	break;
    default:
	pips_internal_error("unexpected style tag");
	break;
    }
    (void) fprintf(stderr,"\n");
}

void print_hpf_dir(void)
{
    (void) fprintf(stderr,"HPF directives:\n");

    print_templates();
    (void) fprintf(stderr,"--------\n");
    print_processors();
    (void) fprintf(stderr,"--------\n");
    print_distributed_arrays();
    (void) fprintf(stderr,"--------\n");
    print_aligns();
    (void) fprintf(stderr,"--------\n");
    print_distributes();
}

void print_templates(void)
{
    (void) fprintf(stderr,"Templates:\n");
    gen_map((gen_iter_func_t)print_entity_variable, list_of_templates());
}

void print_processors(void)
{
    (void) fprintf(stderr,"Processors:\n");
    gen_map((gen_iter_func_t)print_entity_variable, list_of_processors());
}

void print_distributed_arrays(void)
{
    (void) fprintf(stderr,"Distributed Arrays:\n");
    gen_map((gen_iter_func_t)print_entity_variable, list_of_distributed_arrays());
}

void hpfc_print_common(
    FILE *file,
    entity module, 
    entity common)
{
    text t;
    debug_on("PRETTYPRINT_DEBUG_LEVEL");

    t = text_common_declaration(common, module);
    print_text(file, t);
    free_text(t);
    
    debug_off();
}

void hpfc_print_file(string file_name)
{
    string dir_name = db_get_current_workspace_directory();

    safe_system(concatenate("cat ", dir_name, "/", file_name, " >&2", NULL));
}

void fprint_range(
    FILE* file,
    range r)
{
    int lo, up, in;
    bool
	blo = hpfc_integer_constant_expression_p(range_lower(r), &lo),
	bup = hpfc_integer_constant_expression_p(range_upper(r), &up),
	bin = hpfc_integer_constant_expression_p(range_increment(r), &in);

    if (blo && bup && bin)
    {
	if (in==1)
	    if (lo==up)
		fprintf(file, "%d", lo);
	    else
		fprintf(file, "%d:%d", lo, up);
	else
	    fprintf(file, "%d:%d:%d", lo, up, in);
    }
    else
	fprintf(file, "X");
}

void fprint_lrange(
    FILE* file,
    list l)
{
    bool firstrange = true;

    MAP(RANGE, r,
     {
	 if (!firstrange)
	     (void) fprintf(file, ", ");

	 firstrange = false;
	 fprint_range(file, r);
     },
	l);
}

void fprint_message(
    FILE* file,
    message m)
{
    (void) fprintf(file, "message is array %s(", 
		   entity_local_name(message_array(m)));
    fprint_lrange(file, message_content(m));
    (void) fprintf(file, ")\nto\n");
    vect_fprint(file, (Pvecteur) message_neighbour(m), variable_dump_name);
    (void) fprintf(file, "domain is ");
    fprint_lrange(file, message_dom(m));
    (void) fprintf(file, "\n");
}

void fprint_lmessage(
    FILE* file,
    list l)
{
    if (ENDP(l))
	fprintf(file, "message list is empty\n");
    else
	MAP(MESSAGE, m, fprint_message(file, m), l);
}

/*  that is all
 */

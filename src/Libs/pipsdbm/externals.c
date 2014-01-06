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
 * Some functions to manage special (non newgen) resources.
 */

#include "private.h"

#include "linear.h"
#include "ri.h"
#include "complexity_ri.h"
#include "resources.h"

#include "boolean.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"
#include "paf_ri.h"
#include "ri-util.h"

/********************************************************************* UTILS */

/* reads an int while sharing file and buffers with newgen...
 */
static int 
lire_int(FILE * fd)
{
    int c, i = 0, sign = 1;
    genread_in = fd;

    while (isspace(c = genread_input())) ; /* trim */

    if (c == '-') sign = -1;
    else if (isdigit(c))
	i = c-'0';
    else if(c==EOF)
	pips_internal_error("Unexpected EOF, corrupted resource file");
    else
	pips_internal_error("digit or '-' expected : %c %x", c, c);

    while (isdigit(c = genread_input())) 
	i = 10*i + (c-'0');

    return sign*i;
}

/*********************************************************** ENTITY WRAPPERS */

char *
pipsdbm_read_entities(FILE * fd)
{
    int read = gen_read_tabulated(fd, false);
    pips_assert("entities were read", read==entity_domain);
    return (char *) entity_domain;
}

void
pipsdbm_free_entities(char * p)
{
  pips_assert("argument not used", p==p);
  gen_free_tabulated(entity_domain);
}

/**************************************************** OLD STATEMENT MAPPING */

/* The old statement_mapping is an hash_table managed directly.
 * this table associates statement to any newgen type. 
 * Now explicit newgen functions ("->") should be prefered.
 * the storing of the mapping is based on the statement ordering for
 * later reconstruction with the CODE. 
 * tabulating statements would help, but is that desirable?
 *
 * MY opinion is that all newgen objects should have a unique id
 * associated to it, as an addditional hidden field, to support 
 * persistence more easily. Well, there would be some problemes, too.
 *
 * FC.
 */
static int
statement_mapping_count(statement_mapping h)
{
    int n = 0;
    STATEMENT_MAPPING_MAP(s, k,
	if (statement_ordering((statement) s)!=STATEMENT_ORDERING_UNDEFINED) 
	    n++,
	h);
    return n;
}

/** Write a statement mapping.

    This function is quite too low level... It mixes raw printf in a FILE
    with gen_write. To survive other NewGen backend (XML), fprintf could
    be replaced with a gen_fprintf() that could encapsulate the output in
    a CDATA for example in the case of XML.

    But in this case it should be a call to something like
    gen_write_int(fd, order) instead to do even simpler.
*/
void
pipsdbm_write_statement_mapping(
    FILE * fd, /**< file to write to */
    statement_mapping h /**< hash table to dump */)
{
  fprintf(fd, "%d\n", statement_mapping_count(h));
  STATEMENT_MAPPING_MAP(s, v,
  {
    statement key = (statement) s;
    gen_chunkp val = (gen_chunkp) v;
    int order = statement_ordering(key);
    if (order!=STATEMENT_ORDERING_UNDEFINED) { /* save it! */
      fprintf(fd, "%d\n", order);
      gen_write(fd, (gen_chunkp) val);
    } 
    else pips_user_warning("statement with illegal ordering\n");
  },
			h);
}


/** Read a statement mapping.

    This function is quite too low level... It mixes raw getc() from a
    FILE with gen_read. To survive other NewGen backend (XML), fprintf
    could be replaced with a gen_getc() that could peek in a CDATA for
    example in the case of XML.

    But in this case it should be a call to something like so =
    gen_read_int(fd) instead to do even simpler and read an int value (in
    textual form or in <int>...</int> in the case of XML.
*/
hash_table
pipsdbm_read_statement_mapping(FILE * fd)
{
    statement stat;
    hash_table result = hash_table_make(hash_pointer, 0);
    int n;

    pips_assert("some current module name", dbll_current_module);
    pips_debug(3, "statement -> ??? for %s\n", dbll_current_module);
    stat = (statement)
	db_get_memory_resource(DBR_CODE, dbll_current_module, true);

    set_ordering_to_statement(stat);

    /* get meta data.
     */
    n = lire_int(fd);

    while (n-->0) {
	int so = lire_int(fd);
	pips_assert("valid ordering", so!=STATEMENT_ORDERING_UNDEFINED);
	hash_put(result,(void*)ordering_to_statement(so),(void*)gen_read(fd));
    }

	reset_ordering_to_statement();

    return result;
}

/* a little bit partial, because domain are not checked.
 */
bool
pipsdbm_check_statement_mapping(statement_mapping h)
{
    STATEMENT_MAPPING_MAP(k, v, {
	pips_assert("consistent statement", 
		    statement_consistent_p((statement) k));
	pips_assert("consistent val", gen_consistent_p((void*) v));
    },
	h);
    return true;
}

void
pipsdbm_free_statement_mapping(statement_mapping h)
{
    STATEMENT_MAPPING_MAP(k, v, gen_free((void*) v), h);
    FREE_STATEMENT_MAPPING(h);
}

/**************************************** STATEMENT FUNCTIONS: STATEMENT -> */

/* this piece of code makes low level assumptions about newgen internals.
 * see also the comments about statement mappings above.
 */

#define STATEMENT_FUNCTION_MAP(k, v, code, _map_hash_h)			\
    { void * _map_hash_p = NULL;				        \
      void * _map_k; void * _map_v;					\
      while ((_map_hash_p =						\
	   hash_table_scan(_map_hash_h,_map_hash_p,&_map_k,&_map_v))) {	\
        statement k = (statement) ((gen_chunkp)_map_k)->p ;		\
        gen_chunkp v = ((gen_chunkp)_map_v)->p;				\
        code ; }}

/* count the number of statements with a valid ordering. */
static int 
number_of_ordered_statements(hash_table h)
{
    int n = 0;
    STATEMENT_FUNCTION_MAP(s, x, 
    {
      pips_assert("variable not used", x==x);
      if (statement_ordering(s)!=STATEMENT_ORDERING_UNDEFINED) n++;
    }, 
			   h);
    return n;
}

bool
pipsdbm_consistent_statement_function(gen_chunkp map)
{
    hash_table h = (map+1)->h;
    STATEMENT_FUNCTION_MAP(s, x, 
    {
	if (gen_type((void*)s)!=statement_domain) return false;
	if (!gen_consistent_p((void*)x)) return false;
    },
	h);
    return true;
}

/* the stored stuff need be based on the ordering...  because newgen won't
   regenerate pointers...

   Should use a higher level pipsdbm_write_statement_mapping() to survive
   to XML
*/
void
pipsdbm_write_statement_function(
    FILE * fd, /**< file to write to */
    gen_chunkp map /**< statement function */)
{
    hash_table h = (map+1)->h;
    fprintf(fd, "%td\n%d\n", map->i, number_of_ordered_statements(h));
    STATEMENT_FUNCTION_MAP(s, x, 
    {
	int order = statement_ordering(s);
	if (order!=STATEMENT_ORDERING_UNDEFINED) { /* save it! */
	    fprintf(fd, "%d\n", order);
	    gen_write(fd, x);
	} 
	else {
	  pips_user_warning("Statement with illegal ordering, lost data: %s\n",
			    statement_identification(s));
	}
    },
	h);
}


/* Should use a higher level pipsdbm_read_statement_mapping() to survive
   to XML
*/
gen_chunkp
pipsdbm_read_statement_function(FILE * fd /**< file to read from */)
{
    statement stat; 
    int domain, n;
    gen_chunkp result;
    hash_table h;
    
    pips_assert("some current module name", dbll_current_module);
    pips_debug(3, "statement -> ??? for %s\n", dbll_current_module);
    stat = (statement)
	db_get_memory_resource(DBR_CODE, dbll_current_module, true);

    set_ordering_to_statement(stat);

    /* get meta data.
     */
    domain = lire_int(fd);
    n = lire_int(fd);

    /* allocate the statement function. 
     */
    result = gen_alloc(2*sizeof(gen_chunk), GEN_CHECK_ALLOC, domain);
    h = (result+1)->h;

    /* now reads each couple and rebuild the function.
     */
    while(n-->0) {
	int so = lire_int(fd);
	pips_assert("valid ordering", so!=STATEMENT_ORDERING_UNDEFINED);
	HASH_EXTEND(p, p, h, ordering_to_statement(so), gen_read(fd));
    }

    reset_ordering_to_statement();

    return result;
}

/********************************************************************** MISC */

/* Modification Dec 11 1995: ne pas utiliser free_static_control
 * car il libere des champs qui appartiennent a d'autres structures
 * que celles controlees par static_controlize...(champs d'origine)
 * Les liberation de ces champs par un autre transformer (use_def_elim)
 * entrainait alors un core dump au niveau de cette procedure.
 * On fait a la place des gen_free_list en detail --DB ]
 */
 void 
free_static_control_mapping(statement_mapping map)
{
    STATEMENT_MAPPING_MAP(s, val, {
        gen_free_list(static_control_loops((static_control) val));
        gen_free_list(static_control_tests((static_control) val));
        static_control_loops((static_control) val)=NULL;
        static_control_tests((static_control) val)=NULL;
        static_control_params((static_control) val)=NULL;
        gen_free( (void*) val );
    }, map);

    FREE_STATEMENT_MAPPING(map);
}

/* Functions to read and write declarations resource, which is a hash table 
   whose key and value are string (keyword/typedef and TK_keyword/TK_typedef)*/

void declarations_write(FILE * f, hash_table h)
{
  HASH_MAP(k,v,
  {
    fprintf(f, "%s\n", (char *) k);
    fprintf(f, "%td\n", (_int) v);
  },h);
}


hash_table declarations_read(FILE * f) 
{
  hash_table result = hash_table_make(hash_string,0);
  int c;
  while ((c = getc(f)) && c != EOF)
    {
      ungetc(c,f);
      char* key = safe_readline(f);
      _int value = atoi(safe_readline(f));

      hash_put(result,key,(void*)value);
    }
  return result;
}


/*
 * $Id$
 *
 * Some functions to manage special (non newgen) resources.
 */

#include "private.h"

#include "ri.h"
#include "complexity_ri.h"
#include "resources.h"

#include "boolean.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"
#include "ri-util.h"
#include "paf_ri.h"
#include "compsec.h"

char *
pipsdbm_read_entities(FILE * fd)
{
    int read = gen_read_tabulated(fd, FALSE);
    pips_assert("entities were read", read==entity_domain);
    return (char *) entity_domain;
}

void
pipsdbm_free_entities(char * p)
{
    gen_free_tabulated(entity_domain);
}

/* methods_io.c */

/* statement without a proper ordering are not saved on disk */
static int effective_number_of_statements(statement_mapping map)
{
  int n_records = 0;

    STATEMENT_MAPPING_MAP(s, val, {
	statement stat = (statement) s;
	int ordering_of_stat = statement_ordering(stat);
	if(ordering_of_stat != STATEMENT_ORDERING_UNDEFINED) {
	  n_records++;
	}
    }, map);

    pips_assert("write_effects_mapping", 
		STATEMENT_MAPPING_COUNT(map)>=n_records);

    return n_records;
}

extern int genread_input();
extern FILE *genread_in;

static int lire_int(fd)
FILE *fd;
{
    int c;
    int i = 0, sign = 1;

    genread_in = fd;

    while (isspace(c = genread_input())) ;

    if (c == '-') {
	sign = -1;
    }
    else if (isdigit(c)) {
	i = c-'0';
    }
    else {
      if(c==EOF)
	pips_error("lire_int",
		   "Unexpected end of file, corrupted resource file\n");
      else
	pips_error("lire_int", "digit or '-' expected : %c %x\n", c, c);
    }

    while (isdigit(c = genread_input())) {
	i = 10*i + (c-'0');
    }

    return(sign*i);
}

/* statement_effects mappings */
int 
statement_effects_length(statement_effects map)
{
    int n_records = 0;

    STATEMENT_EFFECTS_MAP(s, val, {
	statement stat = (statement) s;
	int ordering_of_stat = statement_ordering(stat);
	if(ordering_of_stat != STATEMENT_ORDERING_UNDEFINED)
	    n_records++;
    }, map);

    return n_records;
}

void
pipsdbm_write_statement_effects(FILE *fd, statement_effects map)
{
    fprintf(fd, "%d\n", statement_effects_length(map));

    STATEMENT_EFFECTS_MAP(s, effs, {
	statement stat = (statement) s;
	int ordering_of_stat = statement_ordering(stat);
	if(ordering_of_stat != STATEMENT_ORDERING_UNDEFINED)
	{
	    fprintf(fd, "%d\n",ordering_of_stat );
	    write_effects(fd, effs);
	}
	else {
	    user_warning("write_effects_mapping",
			 "Statement with illegal ordering, "
			 "effects not stored on disk!\n");
	}
    }, map);
}

statement_effects
pipsdbm_read_statement_effects(FILE *fd)
{
    statement module_stat;
    statement_effects map;
    int ne;

    /* as for writing, loading ordering should be handled at a higher
       level, via tables */
    pips_assert("some current module name", db_get_current_module_name());
    module_stat = (statement) 
	db_get_memory_resource(DBR_CODE, db_get_current_module_name(), TRUE);

    initialize_ordering_to_statement(module_stat);
    map = make_statement_effects();

    ne = lire_int(fd);
    while (ne-- > 0) {
	int ns;
	if((ns = lire_int(fd))==STATEMENT_ORDERING_UNDEFINED)
	    pips_internal_error("Undefined statement ordering\n");

	extend_statement_effects(map, ordering_to_statement(ns), 
				 read_effects(fd));
    }
    
    return(map);
}

void
pipsdbm_free_statement_effects(statement_effects map)
{
    STATEMENT_EFFECTS_MAP(s, val, {
	free_effects((effects) val);
    }, map);
}

void write_static_control_mapping(fd, map)
FILE *fd;
statement_mapping map;
{
    fprintf(fd, "%d\n", effective_number_of_statements(map));

    STATEMENT_MAPPING_MAP(s, val, {
        statement stat = (statement) s;
        int ordering_of_stat = statement_ordering(stat);
	if(ordering_of_stat != STATEMENT_ORDERING_UNDEFINED) {
	  fprintf(fd, "%d\n",ordering_of_stat );
	  write_static_control(fd, (static_control) val );
	} else 
	  pips_user_warning("Statement with illegal ordering\n");
    }, map);
}

/* Modification Dec 11 1995: ne pas utiliser free_static_control */
/* car il libere des champs qui appartiennent a d'autres structures */
/* que celles controlees par static_controlize...(champs d'origine) */
/* Les liberation de ces champs par un autre transformer (use_def_elim) */
/* entrainait alors un core dump au niveau de cette procedure. */
/* On fait a la place des gen_free_list en detail --DB */
 
void free_static_control_mapping(map)
statement_mapping map;
{
    STATEMENT_MAPPING_MAP(s, val, {
        gen_free_list(static_control_loops((static_control) val));
        gen_free_list(static_control_tests((static_control) val));
        static_control_loops((static_control) val)=NULL;
        static_control_tests((static_control) val)=NULL;
        static_control_params((static_control) val)=NULL;
        gen_free( (static_control) val );
    }, map);

    FREE_STATEMENT_MAPPING(map);
}

statement_mapping read_static_control_mapping(fd)
FILE *fd;
{
    statement module_stat;
    statement_mapping map;
    int ne;
    string module_name = db_get_current_module_name();

    pips_assert("current module name", module_name);
    module_stat = (statement)
        db_get_memory_resource(DBR_CODE, module_name, TRUE);
    initialize_ordering_to_statement(module_stat);
    map = MAKE_STATEMENT_MAPPING();

    ne = lire_int(fd);

    while (ne-- > 0) {
        int ns;

	if((ns = lire_int(fd))==STATEMENT_ORDERING_UNDEFINED) {
	  pips_error("read_static_control_mapping",
		     "Undefined statement ordering\n");
	}

        SET_STATEMENT_MAPPING(map,
                              ordering_to_statement(ns),
                              read_static_control(fd));
    }
    return(map);
}

bool check_static_control_mapping(map)
statement_mapping map;
{
    STATEMENT_MAPPING_MAP(s, val, {
        pips_assert("Statement in static control mapping",
		    gen_consistent_p((statement)s));
        pips_assert("Static control mapping ok",
		    gen_consistent_p( (static_control) val));
    }, map);
    return TRUE;
}


void write_transformer_mapping(fd, map)
FILE *fd;
statement_mapping map;
{
    fprintf(fd, "%d\n", effective_number_of_statements(map));

    STATEMENT_MAPPING_MAP(s, val, {
	statement stat = (statement) s;
	int ordering_of_stat = statement_ordering(stat);
	if(ordering_of_stat != STATEMENT_ORDERING_UNDEFINED) {
	  fprintf(fd, "%d\n", ordering_of_stat);
	  write_transformer(fd, (transformer) val);
	} else
	    pips_user_warning("Statement with illegal ordering\n");
    }, map);
}

void print_transformer_mapping(map)
statement_mapping map;
{
    STATEMENT_MAPPING_MAP(s, val, {

	/* The key may be obsolete because the module code
	 * may have been removed, e.g. a transformation was
	 * applied 
	 */
	/* gen_check((statement) s, statement_domain); */
	gen_check((transformer) val, transformer_domain);
	fprintf(stderr,"\t%p\t%p\n", s, val);
    }, map);
}

void free_transformer_mapping(map)
statement_mapping map;
{
    /* print_transformer_mapping(map); */

    STATEMENT_MAPPING_MAP(s, val, {

	/* The key may be obsolete because the module code
	 * may have been removed, e.g. a transformation was
	 * applied 
	 */
	/* gen_check((statement) s, statement_domain); */

	gen_check((transformer) val, transformer_domain);
	free_transformer ((transformer) val);
    }, map);

    FREE_STATEMENT_MAPPING(map);
}

bool check_transformer_mapping(map)
statement_mapping map;
{
    bool consistent_p = TRUE;
    /* Warning: gen_check() aborts on mistake!
     * The following piece of code behaves strangely,
     * but provides key variables to the debugger.
     */

    STATEMENT_MAPPING_MAP(key, val, {
	statement s = (statement) key;
	transformer t = (transformer) val;

	if(!gen_check(s, statement_domain)) {
	    pips_internal_error("Non consistent transformer mapping"
				" for statement key %p\n", s);
	}

	if(!gen_check(t, transformer_domain)) {
	    pips_internal_error("Non consistent transformer mapping"
				" for transformer value %p\n", t);
	}

    }, map);

    return consistent_p;
}

statement_mapping read_transformer_mapping(fd)
FILE *fd;
{
    statement module_stat;
    statement_mapping map;
    string module_name = db_get_current_module_name();
    int ne;

    /* as for writing, loading ordering should be handled at a higher
       level, via tables */
    pips_assert("some current module name", module_name);
    module_stat = (statement) 
	db_get_memory_resource(DBR_CODE, module_name, TRUE);

    initialize_ordering_to_statement(module_stat);
    map = MAKE_STATEMENT_MAPPING();
    ne = lire_int(fd);

    while (ne-- > 0) {
	int ns;
	if((ns = lire_int(fd))==STATEMENT_ORDERING_UNDEFINED)
	    pips_internal_error("Undefined statement ordering\n");

	SET_STATEMENT_MAPPING(map, ordering_to_statement(ns), 
			      read_transformer(fd));
    }

    return(map);
}

void write_complexity_mapping(fd, map)
FILE *fd;
statement_mapping map;
{
    fprintf(fd, "%d\n", effective_number_of_statements(map));

    STATEMENT_MAPPING_MAP(s, val, {
	statement stat = (statement) s;
	int ordering_of_stat = statement_ordering(stat);
	if(ordering_of_stat != STATEMENT_ORDERING_UNDEFINED) {
	  fprintf(fd, "%d\n",ordering_of_stat );
	  write_complexity(fd, (complexity) val);
	} else
	    pips_user_warning("Statement with illegal ordering\n");
    }, map);
}

void free_complexity_mapping(map)
statement_mapping map;
{
    STATEMENT_MAPPING_MAP(s, val, {
	free_complexity((complexity) val);
    }, map);

    FREE_STATEMENT_MAPPING(map);
}

statement_mapping read_complexity_mapping(fd)
FILE *fd;
{
    statement module_stat;
    statement_mapping map;
    string module_name = db_get_current_module_name();
    int ne;

    /* as for writing, loading ordering should be handled at a higher
       level, via tables */
    pips_assert("some current module name", module_name);
    module_stat = (statement) 
	db_get_memory_resource(DBR_CODE, module_name, TRUE);

    initialize_ordering_to_statement(module_stat);

    map = MAKE_STATEMENT_MAPPING();

    ne = lire_int(fd);

    while (ne-- > 0) {
	int ns = lire_int(fd);
	SET_STATEMENT_MAPPING(map, 
			      ordering_to_statement(ns), 
			      read_complexity(fd));
    }

    return(map);
}

void write_compsec_mapping(fd, map)
FILE *fd;
statement_mapping map;
{
    fprintf(fd, "%d\n", effective_number_of_statements(map));

    STATEMENT_MAPPING_MAP(s, val, {
	statement stat = (statement) s;
	int ordering_of_stat = statement_ordering(stat);
	if(ordering_of_stat != STATEMENT_ORDERING_UNDEFINED) {
	  fprintf(fd, "%d\n",ordering_of_stat );
	  write_comp_desc_set(fd, val);
	}
	else {
	    pips_user_warning("Statement with illegal ordering\n");
	}
    }, map);
}

void free_compsec_mapping(map)
statement_mapping map;
{
    STATEMENT_MAPPING_MAP(s, val, {
	free_comp_desc_set((comp_desc_set) val);
    }, map);

    FREE_STATEMENT_MAPPING(map);
}

statement_mapping read_compsec_mapping(fd)
FILE *fd;
{
    statement module_stat;
    statement_mapping map;
    string module_name = db_get_current_module_name();
    int ne;

    /* as for writing, loading ordering should be handled at a higher
       level, via tables */
    pips_assert("some current module name", module_name);
    module_stat = (statement) 
	db_get_memory_resource(DBR_CODE, module_name, TRUE);

    initialize_ordering_to_statement(module_stat);

    map = MAKE_STATEMENT_MAPPING();

    ne = lire_int(fd);

    while (ne-- > 0) {
	int ns;

	if((ns = lire_int(fd))==STATEMENT_ORDERING_UNDEFINED) {
	  pips_error("read_compsec_mapping",
		     "Undefined statement ordering\n");
	}

	SET_STATEMENT_MAPPING(map, 
			      ordering_to_statement(ns), 
			      read_comp_desc_set(fd));
    }

    return(map);
}

bool check_compsec_mapping(map)
statement_mapping map;
{
    STATEMENT_MAPPING_MAP(s, val, 
      {
	  assert(statement_consistent_p((statement) s) &&
		 comp_desc_set_consistent_p((comp_desc_set)val));
      }, map);
    
    return(TRUE);
}

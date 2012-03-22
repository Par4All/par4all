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

/* Some modifications are made to save the current makefile (s.a. files
 * pipsmake/readmakefile.y pipsmake.h )
 *
 * They only occure between following tags: 
 *
 * Bruno Baron
 */

%token PROGRAM
%token MODULE
%token MAIN
%token COMMON
%token TK_CALLEES
%token CALLERS
%token ALL
%token SELECT
%token COMPILATION_UNIT
%token REQUIRED
%token PRODUCED
%token MODIFIED
%token PRESERVED
%token PRE_TRANSFORMATION
%token POST_TRANSFORMATION
%token DOT
%token <name> NAME

%type <name>		phase
%type <name>		resource
%type <owner>		owner
%type <virtual>		virtual
%type <list>		virtuals
%type <rule>		deps
%type <rule>		rule
%type <tag>		dir

%{
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "linear.h"

#include "genC.h"
#include "ri.h"
#include "database.h"
#include "makefile.h"

#include "misc.h"

#include "top-level.h"
#include "pipsmake.h"

#include "stdlib.h"

static makefile pipsmakefile = makefile_undefined;
static hash_table activated = NULL;
%}

%union {
	string name;
	owner owner;
	virtual_resource virtual;
	cons *list;
	rule rule;
	int tag;
}

%%
rules:  	rules rule
		{ add_rule($2); }
	|
		{ 
		    pipsmakefile = make_makefile(NIL, NIL);
            if (activated)
                hash_table_free(activated), activated=NULL;
		    activated = hash_table_make(hash_string, 0);
		}
	;

rule:   	phase deps
		{ rule_phase($2) = $1; $$ = $2; }
	;

deps:		deps dir virtuals
		{
		    if ($2 == REQUIRED) {
			rule_required($1) = 
			    gen_nconc(rule_required($1), $3);
			
		    }
		    else if ($2 == PRODUCED) {
			rule_produced($1) = 
			    gen_nconc(rule_produced($1), $3);
			
		    }
		    else if ($2 == PRESERVED) {
			rule_preserved($1) = 
			    gen_nconc(rule_preserved($1), $3);
		    }
		    else if ($2 == MODIFIED) {
			rule_modified($1) = 
			    gen_nconc(rule_modified($1), $3);
			
		    }
		    else if ($2 == PRE_TRANSFORMATION) {
			rule_pre_transformation($1) = 
			    gen_nconc(rule_pre_transformation($1), $3);
			
		    }
		    else if ($2 == POST_TRANSFORMATION) {
			rule_post_transformation($1) = 
			    gen_nconc(rule_post_transformation($1), $3);
			
		    }
		    else {	
			fprintf(stderr, 
				"[readmakefile] unknown dir: %d\n", $2);
			abort();
		    }
		    
		    $$ = $1;
		}
				
	|
		{ $$ = make_rule(string_undefined, NIL, NIL, NIL, NIL, NIL,NIL); }
	;

dir:		REQUIRED
		{ $$ = REQUIRED; }
	| 	PRODUCED
		{ $$ = PRODUCED; }
	|	MODIFIED
		{ $$ = MODIFIED; }
	|	PRESERVED
		{ $$ = PRESERVED; }
	|	PRE_TRANSFORMATION
		{ $$ = PRE_TRANSFORMATION; }
	|	POST_TRANSFORMATION
		{ $$ = POST_TRANSFORMATION; }
	;

virtuals:	virtuals virtual
		{ $$ = gen_nconc($1, CONS(VIRTUAL_RESOURCE, $2, NIL)); }
	|
		{ $$ = NIL; }
	;

virtual:	owner DOT resource
		{ $$ = make_virtual_resource($3, $1); }
	;

owner:		PROGRAM
		{ $$ = make_owner(is_owner_program, UU); }
	|	MODULE
		{ $$ = make_owner(is_owner_module, UU); }
	|	MAIN
		{ $$ = make_owner(is_owner_main, UU); }
	|	COMMON
		{
		    /*$$ = make_owner(is_owner_common, UU);*/
		    YYERROR;
		}
	|	TK_CALLEES
		{ $$ = make_owner(is_owner_callees, UU); }
	|	CALLERS
		{ $$ = make_owner(is_owner_callers, UU); }
	|	ALL
		{ $$ = make_owner(is_owner_all, UU); }
	|	SELECT
		{ $$ = make_owner(is_owner_select, UU); }
	|	COMPILATION_UNIT
		{ $$ = make_owner(is_owner_compilation_unit, UU); }
	;

phase:		NAME
		{ $$ = strupper($1, $1); }
	;

resource:	NAME
		{ $$ = strupper($1, $1); }
	;
%%

void yyerror_lex_part(const char *);
void yyerror(const char * s)
{
    int c;
    yyerror_lex_part(s);
    fprintf(stderr, "[readmakefile] unparsed text:\n");
    while ((c = getc(yyin)) != EOF) putc(c, stderr);
    exit(EXIT_FAILURE);
}

void
fprint_virtual_resources(FILE *fd, const char* dir, list lrv)
{
    MAP(VIRTUAL_RESOURCE, vr,
    {
	owner o = virtual_resource_owner(vr);
	string n = virtual_resource_name(vr);
	
	switch (owner_tag(o)) {
	case is_owner_program:
	    fprintf(fd, "    %s program.%s\n", dir, n);
	    break;
	case is_owner_module:
	    fprintf(fd, "    %s module.%s\n", dir, n);
	    break;
	case is_owner_main:
	    fprintf(fd, "    %s main.%s\n", dir, n);
	    break;
	case is_owner_callees:
	    fprintf(fd, "    %s callees.%s\n", dir, n);
	    break;
	case is_owner_callers:
	    fprintf(fd, "    %s callers.%s\n", dir, n);
	    break;
	case is_owner_all:
	    fprintf(fd, "    %s all.%s\n", dir, n);
	    break;
	case is_owner_select:
	    fprintf(fd, "    %s select.%s\n", dir, n);
	    break;
	case is_owner_compilation_unit:
	    fprintf(fd, "    %s compilation_unit.%s\n", dir, n);
	    break;
	default:
	    pips_internal_error("bad owner tag (%d)\n", owner_tag(o));
	}
    }, lrv);
}

void fprint_makefile(FILE *fd, makefile m)
{
    MAP(RULE, r, {
	fprintf(fd, "%s\n", rule_phase(r));
	fprint_virtual_resources(fd, "\t!", rule_pre_transformation(r));
	fprint_virtual_resources(fd, "\t<", rule_required(r));
	fprint_virtual_resources(fd, "\t>", rule_produced(r));
	fprint_virtual_resources(fd, "\t=", rule_preserved(r));
	fprint_virtual_resources(fd, "\t#", rule_modified(r));
    }, makefile_rules(m));
}
		
makefile 
parse_makefile(void)
{
    if (pipsmakefile == makefile_undefined)
    {
	debug_on("PIPSMAKE_DEBUG_LEVEL");

	yyin = fopen_config(PIPSMAKE_RC, NULL, "PIPS_PIPSMAKERC");

	init_lex();
	yyparse();
	safe_fclose(yyin, "PIPS_PIPSMAKERC");
	
	ifdebug(8) fprint_makefile(stderr, pipsmakefile);

	debug_off();
    }

    return pipsmakefile;
}


/* this function returns the rule that defines builder pname */
rule find_rule_by_phase(const char* pname)
{
  rule fr = rule_undefined;
  makefile m = parse_makefile();

  pips_debug(9, "searching rule for phase %s\n", pname);

  /* walking thru rules */
  FOREACH(RULE, r, makefile_rules(m)) {
    if (same_string_p(rule_phase(r), pname)) {
      fr = r;
      break;
    }
  }

  return fr;
}

void add_rule(rule r)
{
    string pn = rule_phase(r);
    bool   active_phase = false;

    /* Check resources produced by this rule */
    FOREACH(VIRTUAL_RESOURCE, vr, rule_produced(r)) {
	string vrn = virtual_resource_name(vr);
	string phase;

	/* We activated this rule to produce this resource only if
	 * this resource is not already produced */
	if ((phase = hash_get(activated, vrn)) == HASH_UNDEFINED_VALUE) {
	    pips_debug(1, "Default function for %s is %s\n", vrn, pn);

	    active_phase = true;
	    hash_put(activated, vrn, pn);

	    makefile_active_phases(pipsmakefile) = 
		CONS(STRING, strdup(pn), makefile_active_phases(pipsmakefile));
	}
	else debug(1, "add_rule", "Function %s not activated\n", pn);
    }

    /* Check resources required for this rule if it is an active one */
    if (active_phase) {
	FOREACH(VIRTUAL_RESOURCE, vr, rule_required(r)) {
	    string vrn = virtual_resource_name(vr);
	    owner vro = virtual_resource_owner(vr);
	    string phase;

	    /* We must use a resource already defined */
	    if ( owner_callers_p(vro) || owner_callees_p(vro) ) {}
	    else {
		phase = hash_get(activated, vrn);
		if (phase == HASH_UNDEFINED_VALUE) {
		    user_warning( "add_rule",
				  "%s: phase %s requires an undefined resource %s\n",
				  PIPSMAKE_RC, pn, vrn);
		}
		/* If we use a resource, another function should have produced it */
		else if (strcmp(phase, pn) == 0) {
		    pips_internal_error(
			       "%s: phase %s cannot be active for the %s resource\n",
			       PIPSMAKE_RC, phase, vrn);
		}
		else pips_debug(1,
			   "Required resource %s is checked OK for Function %s\n",
			   vrn, pn);
	    }
	}
    }

    /* FI: no check of rule name unicity; e.g. double declaration of sesamify */
    rule or = find_rule_by_phase(rule_phase(r));
    if(rule_undefined_p(or))
      makefile_rules(pipsmakefile) = gen_nconc(makefile_rules(pipsmakefile),
					       CONS(RULE, r, NIL));
    else
      pips_internal_error("Double declaration of phase \"%s\"\n"
			  "Check pipsmake-rc.tex or pipsmake.rc\n", rule_phase(r));
}

makefile open_makefile(const char* name)
{
    FILE * fd;
    char * mkf_name;

    mkf_name = build_pgm_makefile(name);
    fd = fopen(mkf_name, "r");

    if (!makefile_undefined_p(pipsmakefile)) 
    {
      free_makefile(pipsmakefile);
      pipsmakefile = makefile_undefined;
      pips_debug(1, "current makefile erased\n");
    }

    if (fd)
    {
      pipsmakefile = read_makefile(fd);
      safe_fclose(fd, mkf_name);
    }
    else
    {
      pips_user_warning("pipsmake file not found in database...\n");
      pipsmakefile = parse_makefile();
    }

    free(mkf_name);
    return pipsmakefile;
}

void 
save_makefile(const char* name)
{
    char * mkf_name = build_pgm_makefile(name);
    FILE * fd = safe_fopen(mkf_name, "w");
    write_makefile(fd, pipsmakefile);
    safe_fclose(fd, mkf_name);
    pips_debug(1, "makefile written on %s\n", mkf_name);
    free(mkf_name);
}

bool 
close_makefile(const char* name)
{
    save_makefile(name);
    free_makefile(pipsmakefile), pipsmakefile = makefile_undefined;
    return true;
}

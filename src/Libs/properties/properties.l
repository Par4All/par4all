%{
/* The syntax of a property list. */

#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "properties.h"
#include "misc.h"
#include "top-level.h"

#define TTRUE	10
#define TFALSE	11
#define TIDENT   12
#define TNUMB	13
#define TSTRING  14

static hash_table pl = (hash_table) NULL;

%}

%%
TRUE                    {return(TTRUE);}
FALSE			{return(TFALSE);}
[-_a-zA-Z][-_a-zA-Z0-9]* {return(TIDENT);}
[0-9]+       		{return(TNUMB);}
\"[^\"]*\"		{return(TSTRING);}
^[ \t]*#.*$		;
[ \t\n]*		;
.                       {fprintf(stderr, 
			         "[read_properties] unknown char %c (%d)\n", 
				 *yytext, *yytext);
			 exit(1);
			}
%%
yywrap()
{
    return(1) ;
}

static void parse_properties()
{
    int tok;
    property pr;
    property opr;
    
    while (tok = yylex()) {
	string n;
	property p;
	
	if (tok != TIDENT) {
	    fprintf(stderr, 
		    "[read_properties] syntax error in property file\n");
	    exit(1);
	}
			  
	n = strdup(yytext);

	switch (tok = yylex()) {
	  case TTRUE:
	    pr = make_property(is_property_bool, TRUE);
	    break;
	  case TFALSE:
	    pr = make_property(is_property_bool, FALSE);
	    break;
	  case TNUMB:
	    pr = make_property(is_property_int, atoi(yytext));
	    break;
	  case TSTRING: {
            char *s = strdup(yytext);
            char *q = strrchr(s, '"');
            if (q == NULL) {
                fprintf(stderr,"[read_properties] bad string : %s\n", yytext);
	        exit(1);
            }
            *q = '\0';            
	    pr = make_property(is_property_string, s+1);
	    break;
          }
	  default:
	    fprintf(stderr, 
		    "[read_properties] bad value for property %s : %s\n", 
		    n, yytext);
	    exit(1);
	}

	if((opr = hash_get(pl, n)) != HASH_UNDEFINED_VALUE) {
            user_warning("parse_properties","property %s redefined\n", n);
            free_property(opr);
            hash_update(pl, n, (char *) pr); 
        }
        else
	    hash_put(pl, n, (char *) pr);
    }
}

static void read_properties()
{
    if (pl != (hash_table) NULL)
	return;

    pl = hash_table_make(hash_string, 0);

    if ((yyin = fopen(PROPERTIES_LIB_FILE, "r")) == NULL) {
	pips_error("read_properties", "no default properties\n");
    }
    else
	parse_properties();

    if ((yyin = fopen(PROPERTIES_FILE, "r")) != NULL) {
	parse_properties();
    }
}

static property get_property(name)
string name;
{
    property p;

    if (pl == (hash_table) NULL)
	read_properties();

    if ((p=(property) hash_get(pl, name)) == (property)HASH_UNDEFINED_VALUE) {
	fprintf(stderr, "[get_property] unknown property : %s\n", name);
	exit(1);
    }

    return(p);
}

bool get_bool_property(name)
string name;
{
    property p = get_property(name);

    if (! property_bool_p(p)) {
	fprintf(stderr, "[get_bool_property] property %s is not bool\n", name);
	exit(1);
    }

    return(property_bool(p));
}

void set_bool_property(name, b)
string name;
bool b;
{
    property p = get_property(name);

    if (! property_bool_p(p)) {
	fprintf(stderr, "[set_bool_property] property %s is not bool\n", name);
	exit(1);
    }

    property_bool(p) = b;
}

string get_string_property(name)
string name;
{
    property p = get_property(name);

    if (! property_string_p(p)) {
	fprintf(stderr, "[get_string_property] property %s is not string\n", 
		name);
	exit(1);
    }

    return(property_string(p));
}

void set_string_property(name, s)
string name;
string s;
{
    property p = get_property(name);

    if (! property_string_p(p)) {
	fprintf(stderr, "[get_string_property] property %s is not string\n", 
		name);
	exit(1);
    }

    free(property_string(p));
    property_string(p) = strdup(s);
}

int get_int_property(name)
string name;
{
    property p = get_property(name);

    if (! property_int_p(p)) {
	fprintf(stderr, "[get_int_property] property %s is not int\n", 
		name);
	exit(1);
    }

    return(property_int(p));
}

void set_int_property(name, i)
string name;
int i;
{
    property p = get_property(name);

    if (! property_int_p(p)) {
	fprintf(stderr, "[get_int_property] property %s is not int\n", 
		name);
	exit(1);
    }

    property_int(p) = i;
}

static string true( b )
bool b ;
{
    return( (b == TRUE) ? "TRUE" : "FALSE" ) ;
}

void fprint_properties(fd)
FILE *fd;
{
    if (pl == (hash_table) NULL)
	read_properties();

    HASH_MAP(n, p, {
	property pr = (property) p;
	fprintf(fd, "%40.40s ", n);
	if (property_bool_p(pr)) {
	    fprintf(fd, "%s\n", true(property_bool(pr)));
	}
	else if (property_int_p(pr)) {
	    fprintf(fd, "%d\n", property_int(pr));
	}
	else if (property_string_p(pr)) {
	    fprintf(fd, "%s\n", property_string(pr));
	}
	else {
	    fprintf(fd, "[fprint_properties] property %s : bad type\n", n); 
	    abort();
	}
    }, pl);
}

/* $RCSfile: properties.l,v $ (version $Revision$)
 * $Date: 1997/03/07 14:14:31 $, 
 * 
 * - 01/96: PIPS_PROPERTIESRC added. FC.
 */

%{
/* The syntax of a property list. */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

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

#ifdef FLEX_SCANNER

/* We may parse strings or files...
 */
static char * string_to_parse = (char*) 0; /* shared pointer! */

#define YY_INPUT(buffer, result, max_size)		\
{							\
    int i=0;						\
    if (string_to_parse) /* we're parsing a string */	\
    {							\
	while(i<max_size && *string_to_parse)		\
	    buffer[i++]=*string_to_parse++;		\
    }							\
    else /* it's a file */				\
    {							\
	int c;						\
	while (i<max_size && (c=getc(yyin))!=EOF)	\
	    buffer[i++]=(char) c;			\
    }							\
    result = i==0? YY_NULL: i;				\
}

static void parse_properties(void);

void parse_properties_string(char *s) 
{
    string_to_parse = s;
    parse_properties();
    string_to_parse = (char *) 0;
}

#else /* ATT/POSIX just seem to like parsing yyin... */

void parse_properties_string(char *s)
{
    pips_internal_error("cannot parse string (%s) without flex\n", s);
}

#endif /* FLEX_SCANNER */

%}

%%
TRUE                    {return(TTRUE);}
FALSE			{return(TFALSE);}
[-_a-zA-Z][-_a-zA-Z0-9]* {return(TIDENT);}
[0-9]+       		{return(TNUMB);}
\"[^\"]*\"		{return(TSTRING);}
^[ \t]*#.*$		;
[ \t\n]*		;
=			; /* ignore an = */
.                       {fprintf(stderr, 
			         "[read_properties] unknown char %c (%d)\n", 
				 *yytext, *yytext);
			 exit(1);
			}
%%

int yywrap(void)
{ return 1;}

static void parse_properties(void)
{
    int tok;
    property pr;
    property opr;
    
    while ((tok = yylex())) {
	string n;
	
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
	      /* GO: We need to free(s) so the skeep now 
		 the quote and do not use s+1 anymore*/
            char *s = strdup(yytext + 1);
            char *q = strrchr(s, '"');
            if (q == NULL) {
                fprintf(stderr,"[read_properties] bad string : %s\n", yytext);
	        exit(1);
            }
            *q = '\0';            
	    pr = make_property(is_property_string, s);
	    break;
          }
	  default:
	    fprintf(stderr, 
		    "[read_properties] bad value for property %s : %s\n", 
		    n, yytext);
	    exit(1);
	}

	if ((opr = (property) hash_get(pl, n)) != 
            (property) HASH_UNDEFINED_VALUE) {
            if(!property_equal_p(pr, opr)) {
                user_warning("parse_properties","property %s redefined\n", n);
            }
            free_property(opr);
            hash_update(pl, n, (char *) pr); 
        }
        else
	    hash_put(pl, n, (char *) pr);
    }
}

bool property_equal_p(property p1, property p2)
{
    bool equal_p = (property_tag(p1)==property_tag(p2));

    if(equal_p) {
        switch(property_tag(p1)) {
        case is_property_int:
            equal_p = (property_int(p1)==property_int(p2));
            break;
        case is_property_bool:
            equal_p = (property_bool(p1)==property_bool(p2));
            break;
        case is_property_string:
            equal_p = (strcmp(property_string(p1),property_string(p2))==0);
            break;
        default:
            pips_error("property_equal_p", "ill. tag %d\n", property_tag(p1));
        }
    }

    return equal_p;
}

static void read_properties()
{
    string rc;

    if (pl != (hash_table) NULL)
	return;
    else
	pl = hash_table_make(hash_string, 0);

    /* ${PIPS_ROOT}/Share/properties.rc
     */
    rc = PROPERTIES_LIB_FILE;
    if ((yyin = fopen(rc, "r")) == NULL)
	user_warning("read_properties",
                   "no default properties (%s), check PIPS installation!\n", rc);
    else
	parse_properties();
    free(rc);

    /* ${PIPS_PROPERTIESRC}
     */
    rc = getenv("PIPS_PROPERTIESRC");
    if (rc)
    {
	if ((yyin = fopen(rc, "r")) == NULL) {
	    fprintf(stderr, "[read_properties] "
            "no properties (%s), check Shell variable $PIPS_PROPERTIESRC\n", rc);
            exit(1);
        }
	else
	    parse_properties();
    }
    
    /* ./properties.rc
     */
    if ((yyin = fopen(PROPERTIES_FILE, "r")) != NULL)
	parse_properties();
}

property get_property(name)
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

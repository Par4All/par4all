%{
/*
 * $Id$
 * 
 * - 01/1996: PIPS_PROPERTIESRC added. FC.
 * - 09/1997: save/load properties in database. FC.
 */

/* The syntax of a property list. */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "genC.h"
#include "properties.h"
#include "misc.h"
#include "constants.h"

#define TTRUE	10
#define TFALSE	11
#define TIDENT   12
#define TNUMB	13
#define TSTRING  14
#define ENDOFLINE 15

/* properties are stored in this hash table (string -> property)
 * for fast accesses.
 */
static hash_table pl = (hash_table) NULL;
static bool update_property = FALSE;

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
    update_property = TRUE;
    string_to_parse = s;
    parse_properties();
    string_to_parse = (char *) 0;
    update_property = FALSE;
}

#else /* ATT/POSIX just seem to like parsing yyin... */

void parse_properties_string(char *s)
{
    fprintf(stderr, "[parse_properties_string] "
	    "cannot parse string (%s) without flex\n", s);
    exit(1);
}

#endif /* FLEX_SCANNER */

%}

%%
TRUE                    { return(TTRUE); }
FALSE			{ return(TFALSE); }
[0-9]+       		{ return(TNUMB); }
[-_a-zA-Z][-_a-zA-Z0-9]* { return(TIDENT); }
\"[^\"]*\"		{ /*"*/return(TSTRING); }
^[ \t]*#.*$		{ /* comments are skipped. */ }
[ \t\n]*		{ /* blanks are skipped. */ }
=			{ /* ignore an = */ }
.                       { fprintf(stderr, "skipping unexpected char %c (%d)\n",
				  *yytext, *yytext);
			}
%%

#define error(f, s)  fprintf(stderr, "[property] " f, s), abort()

int yywrap(void)
{ return 1;}

static bool 
property_equal_p(property p1, property p2)
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
	    error("illegal tag %d\n", property_tag(p1));
        }
    }

    return equal_p;
}

static string property_tag_to_string(tag t)
{
    switch (t) {
    case is_property_int: return "int";
    case is_property_bool: return "bool";
    case is_property_string: return "string";
    default: return "UNKNOW property type";
    } 
}

static property get_property(string, bool);

static void 
parse_properties(void)
{
    int tok;
    property pr = property_undefined, opr;
    
    if (!pl) pl = hash_table_make(hash_string, 0); /* lazy init */

    while ((tok = yylex())) 
    {
	string n;
	
	if (tok != TIDENT) {
	    fprintf(stderr, "syntax error in parse_properties, "
		    "ident expected, got %d for -%s-.\n", tok, yytext);
	    return;
	}

	n = strdup(yytext);

	if (update_property && property_undefined_p(get_property(n, TRUE))) {
	    fprintf(stderr, "user error in parse_properties: "
		    "Unknown property %s to update\n", n);
	    return;
	}

	switch (tok = yylex()) 
	{
	case TTRUE:
	    pr = make_property(is_property_bool, (void*) TRUE);
	    break;
	case TFALSE:
	    pr = make_property(is_property_bool, (void*) FALSE);
	    break;
	case TNUMB:
	    pr = make_property(is_property_int, (void*) atoi(yytext));
	    break;
	case TSTRING: {
	    /* GO: We need to free(s) so the skeep now 
	       the quote and do not use s+1 anymore*/
            char *s = strdup(yytext + 1);
            char *q = strrchr(s, '"');
            if (q == NULL) {
		fprintf(stderr, "user error in parse_properties: "
			"bad string : %s\n", yytext);
		return;
	    }
            *q = '\0';            
	    pr = make_property(is_property_string, s);
	    break;
	}
	default:
	    fprintf(stderr, "user error in parse_properties: "
		    "bad value for property %s: %s\n", n, yytext);
	    return;
	}

	if ((opr = (property) hash_get(pl, n)) != 
            (property) HASH_UNDEFINED_VALUE) 
	{
	    if (property_tag(opr)!=property_tag(pr))
	    {
		fprintf(stderr, "user warning, "
			"cannot change type from %s to %s "
			"when updating property %s\n",
			property_tag_to_string(property_tag(opr)),
			property_tag_to_string(property_tag(pr)),
			n);
		free_property(pr);
	    }
	    else 
	    {
		if(!property_equal_p(pr, opr))
		    fprintf(stderr, "user warning, "
			    "property %s redefined\n", n);
		free_property(opr);
		hash_update(pl, n, (char *) pr);
	    }
        }
        else
	    hash_put(pl, n, (char *) pr);
    }
}

static void
parse_properties_file(FILE * fd)
{
    yyrestart(fd);
    parse_properties();
}

static void 
read_properties(void)
{
    string rc;

    if (pl != (hash_table) NULL)
	return;
    /* else */
    pl = hash_table_make(hash_string, 0);
    update_property = FALSE;

    /* ${PIPS_ROOT}/Share/properties.rc
     */
    rc = PROPERTIES_LIB_FILE;
    if ((yyin = fopen(rc, "r")) == NULL) {
	fprintf(stderr, "[read_properties] no default properties (%s), "
		"check PIPS installation!\n", rc);
    } else {
	parse_properties_file(yyin);
	fclose(yyin), yyin = (FILE*) NULL;
    }
    free(rc);

    /* ${PIPS_PROPERTIESRC}
     */
    rc = getenv("PIPS_PROPERTIESRC");
    if (rc)
    {
	if ((yyin = fopen(rc, "r")) == NULL)
	    fprintf(stderr, "[read_properties] no properties (%s), "
		    "check Shell variable $PIPS_PROPERTIESRC\n", rc);
	/* else */
	parse_properties_file(yyin);
	fclose(yyin), yyin = (FILE*) NULL;
    }
    
    /* ./properties.rc
     */
    if ((yyin = fopen(PROPERTIES_FILE, "r")) != NULL) {
	parse_properties_file(yyin);
	fclose(yyin), yyin = (FILE*) NULL;
    }
}

static property
get_property(string name, bool cool)
{
    property p;
    if (!pl) read_properties(); /* rather lazy... */
    if ((p=(property) hash_get(pl, name)) == (property)HASH_UNDEFINED_VALUE) {
	if (!cool) error("unknown property: --%s--\n", name);
	/* else */
	p = property_undefined;
    }
    return p;
}

bool 
get_bool_property(string name)
{
    property p = get_property(name, FALSE);
    if (! property_bool_p(p))
	error("property %s is not bool\n", name);
    return(property_bool(p));
}

void 
set_bool_property(string name, bool b)
{
    property p = get_property(name, FALSE);
    if (! property_bool_p(p))
	error("property %s is not bool\n", name);
    property_bool(p) = b;
}

string 
get_string_property(string name)
{
    property p = get_property(name, FALSE);
    if (! property_string_p(p)) 
	error("property %s is not string\n", name);
    return property_string(p);
}

void 
set_string_property(string name, string s)
{
    property p = get_property(name, FALSE);
    if (! property_string_p(p))
	error("property %s is not string\n", name);
    free(property_string(p));
    property_string(p) = strdup(s);
}

int 
get_int_property(string name)
{
    property p = get_property(name, FALSE);
    if (! property_int_p(p)) 
	error("property %s is not int\n", name);
    return property_int(p);
}

void 
set_int_property(string name, int i)
{
    property p = get_property(name, FALSE);
    if (! property_int_p(p)) 
	error("property %s is not int\n", name);
    property_int(p) = i;
}

void fprint_property_direct(FILE * fd, string pname)
{
    property p = get_property(pname, TRUE);
    if (property_undefined_p(p)) 
	fprintf(fd, "# undefined property %s\n", pname);
    else
	switch (property_tag(p)) {
	case is_property_bool: 
	    fprintf(fd, "%s", property_bool(p)? "TRUE": "FALSE");
	    break;
	case is_property_int:
	    fprintf(fd, "%d", property_int(p));
	    break;
	case is_property_string:
	    fprintf(fd, "%s", property_string(p));
	    break;
	default: 
	    fprintf(fd, "# bad property type for %s\n", pname);
	}
    
}

void 
fprint_property(FILE * fd, string pname)
{
    property p = get_property(pname, TRUE);
    if (property_undefined_p(p)) 
	fprintf(fd, "# undefined property %s\n", pname);
    else
	switch (property_tag(p)) {
	case is_property_bool: 
	    fprintf(fd, "%s %s\n", pname, property_bool(p)? "TRUE": "FALSE");
	    break;
	case is_property_int:
	    fprintf(fd, "%s %d\n", pname, property_int(p));
	    break;
	case is_property_string:
	    fprintf(fd, "%s \"%s\"\n", pname, property_string(p));
	    break;
	default: 
	    fprintf(fd, "# bad property type for %s\n", pname);
	}
}

void 
fprint_properties(FILE * fd)
{
    if (pl) 
    { /* dump all sorted properties in fd. */
	gen_array_t array = gen_array_make(0);
	fprintf(fd, "# PIPS PROPERTIES\n");
	HASH_MAP(name, p, gen_array_dupappend(array, name), pl);
	gen_array_sort(array);
	GEN_ARRAY_MAP(name, fprint_property(fd, name), array);
	gen_array_full_free(array);
    } else fprintf(fd, "# NO PIPS PROPERTIES...\n");
}

/******************************************************* TOP-LEVEL INTERFACE */

extern string db_get_meta_data_directory(void); /* in pipsdbm */

static string get_property_file_name(void)
{
    string dir_name = db_get_meta_data_directory(),
	name = strdup(concatenate(dir_name, "/properties", 0));
    free(dir_name); return name;
}

/* when opening a workspace, retrieve the properties.
 * @return whether okay
 */
bool open_properties(void)
{
    string file_name = get_property_file_name();
    FILE * file = check_fopen(file_name, "r");
    if (!file) return FALSE;
    if (pl) hash_table_free(pl), pl = (hash_table) NULL; /* lazy clean... */
    parse_properties_file(file);
    safe_fclose(file, file_name);
    free(file_name);
    return TRUE;
}

/* on close, save the properties in the current workspace.
 * this is called from close_workspace, so there is some current one.
 */
void save_properties(void)
{
    string file_name = get_property_file_name();
    FILE * file = safe_fopen(file_name, "w");
    fprint_properties(file);
    safe_fclose(file, file_name);
    free(file_name);
}

/* the creation is demand driven from get_property...
 */

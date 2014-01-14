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
%option nounput
%option noinput

%{
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
/*
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
#include "top-level.h"
#include "constants.h"
#include "pipsdbm.h"

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
static bool update_property = false;

/* A flag used to avoid infinite recursion in the case of use error here */
static size_t pending_errors = false;


/* Call pips_user_error and notice it first */	\
#define property_user_error(...) do {		\
  pending_errors++;				\
  user_error(__FUNCTION__, __VA_ARGS__);	\
} while(0)

#ifdef FLEX_SCANNER

/* We may parse strings or files...
 */
static char * string_to_parse = (char*) 0; /* shared pointer! */

#define YY_INPUT(buffer, result, max_size)		\
{							\
  unsigned int i = 0;					\
  if (string_to_parse) /* we're parsing a string */	\
  {							\
    while (i<max_size && *string_to_parse)		\
      buffer[i++] = *string_to_parse++;			\
  }							\
  else /* it's a file */				\
  {							\
    int c;						\
    while (i<max_size && (c=getc(yyin))!=EOF)		\
      buffer[i++] = (char) c;				\
  }							\
  result = i==0? YY_NULL: i;				\
}

static void parse_properties(void);

void parse_properties_string(char *s)
{
    yyrestart(NULL); // In case, lex has been interrupted by an error
    update_property = true;
    string_to_parse = s;
    parse_properties();
    string_to_parse = (char *) 0;
    update_property = false;
}

#else /* ATT/POSIX just seem to like parsing yyin... */

void parse_properties_string(char *s)
{
    property_user_error("cannot parse string (%s) without flex\n", s);
}

#endif /* FLEX_SCANNER */

%}

%%
TRUE                    { return(TTRUE); }
FALSE			{ return(TFALSE); }
-?[0-9]+       		{ return(TNUMB); }
[-_a-zA-Z][-_a-zA-Z0-9]* { return(TIDENT); }
\"[^\"]*\"		{ return(TSTRING); }
^[ \t]*#.*$		{ /* comments are skipped. */ }
[ \t\n]*		{ /* blanks are skipped. */ }
=			{ /* ignore =: foo=bla same as foo bla */ }
.                       { fprintf(stderr, "skipping unexpected char %c (%d)\n",
				  *yytext, *yytext);
			}
%%

int yywrap(void)
{ return 1;}

static bool property_equal_p(property p1, property p2)
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
	    property_user_error("illegal tag %d\n", property_tag(p1));
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

static void parse_properties(void)
{
    int tok;
    property pr = property_undefined, opr;

    if (!pl) pl = hash_table_make(hash_string, 0); /* lazy init */

    while ((tok = yylex()))
    {
	string n;

	if (tok != TIDENT) {
	    property_user_error("syntax error: ident expected, got %d for -%s-.\n", tok, yytext);
	    return;
	}

	n = strdup(yytext);

	if (update_property && property_undefined_p(get_property(n, true))) {
	  property_user_error("Unknown property \"%s\" to update\n", n);
	  free(n);
	  return;
	}

	switch (tok = yylex())
	{
	case TTRUE:
	    pr = make_property_bool(true);
	    break;
	case TFALSE:
	    pr = make_property_bool(false);
	    break;
	case TNUMB:
	    pr = make_property_int(atol(yytext));
 	    break;
	case TSTRING: {
	    /* GO: We need to free(s) so the skeep now
	       the quote and do not use s+1 anymore*/
            char *s = strdup(yytext + 1);
            char *q = strrchr(s, '"'); // "
			if (q == NULL) {
				property_user_error("bad string : %s\n", yytext);
				free(n);
				return;
			}
            *q = '\0';
	    pr = make_property_string(s);
	    break;
	}
	default:
	    property_user_error("bad value for property %s: %s\n", n, yytext);
	    return;
	}

	if ((opr = (property) hash_get(pl, n)) !=
            (property) HASH_UNDEFINED_VALUE)
	{
	    if (property_tag(opr)!=property_tag(pr))
	    {
	      property_user_error("cannot change type from %s to %s "
				  "when updating property %s\n",
				  property_tag_to_string(property_tag(opr)),
				  property_tag_to_string(property_tag(pr)),
				  n);
	      free_property(pr);
	    }
	    else
	    {
	      hash_update(pl, n, (char *) pr);
	      if(!property_equal_p(pr, opr))
      		pips_debug(1,"property %s redefined\n", n);
	      free_property(opr);
	    }
        }
        else
	    hash_put(pl, n, (char *) pr);
		free(n);
    }
}

static void parse_properties_file(FILE * fd)
{
    string_to_parse = NULL;
    update_property = false; // FI: Tpips/property.tpips, to survive
			     // the open after the close
    yyrestart(fd);
    parse_properties();
}

static void read_properties(void)
{
    FILE * old_prop = NULL;

    if (pl != (hash_table) NULL)
    return;
    /* else */
    pl = hash_table_make(hash_string, 0);
    update_property = false;

    /* Default properties:
     *
     * if the variable PIPS_PROPERTIESRC is defined in the environment,
     * it contains the absolute property file name to use;
     *
     * else: ${PIPS_ROOT}/etc/properties.rc
     */
    if( (prop_in = fopen_config(PROPERTIES_RC,NULL,"PIPS_PROPERTIESRC")) )
    {
        parse_properties_file(prop_in);
        fclose(prop_in), prop_in = (FILE*) NULL;
    }

    /* An older version of properties.rc might be useful for the current
     * PIPS execution. This is used to maintain backward compatibility
     * for old non-regression tests.
     */
    string pfn = string_undefined;
    string sdn = get_script_directory_name();
    pfn = strdup(concatenate(sdn, "/", OLD_PROPERTIES_RC, NULL));
    if ((old_prop = fopen(pfn, "r")) != NULL) {
      char opfn[100];
      int stat = fscanf(old_prop, "%s", opfn);
      assert(stat != EOF);

      if ((prop_in = fopen_config(opfn, NULL, NULL)) != NULL) {
        parse_properties_file(prop_in);
        fclose(prop_in), prop_in = (FILE*) NULL;
      }
      fclose(old_prop), old_prop = (FILE*) NULL;
    }
    free(pfn);

    /* Update the standard properties with a locally defined ./properties.rc
     * Is . the cwd or the directory where the .tpips script has been found?
     */
    if ((prop_in = fopen(PROPERTIES_RC, "r")) != NULL) {
    parse_properties_file(prop_in);
    fclose(prop_in), prop_in = (FILE*) NULL;
    }
}


/* Test if we have too many property errors pending

   It is useful in pips_user_error to avoid infinite recursion where
   properties are used there to change the behaviour of pips_user_error...
   */
bool too_many_property_errors_pending_p() {
  /* Well, this limit should depend on the actual implementation of
     user_error, that may change in tpips, pyps, wpips... It should be
     configurable... */
  return pending_errors > 3;
}


/* Reset the pending stuff in case we where able to cope with... */
void reset_property_error() {
  pending_errors = 0;
}


/* Get a property

   @param[in] name is the name of the property to look-up

   @param[in] cool is a bool setting the behaviour if the property does
   not exist. If the property does not exist and cool is true, it return
   property_undefined but if false it aborts

   @return the property or abort() if it does not exist and cool is false
*/
property get_property(const char* name, bool cool)
{
    property p = property_undefined;
    property u = (property) HASH_UNDEFINED_VALUE;

    if (!pl)
       read_properties(); /* rather lazy... */

    p = (property) hash_get(pl, name);

    if (p == u) {
	if (!cool) {
	  property_user_error("unknown property: --%s--\n", name);
        }
	/* else since error does not return */
	p = property_undefined;
    }
    return p;
}


/* Get the value of a bool property

   @param[in] name is the name of the property to look-up

   @return the bool value
*/
bool get_bool_property(const char* name)
{
    property p = get_property(name, false);
    if (! property_bool_p(p))
	property_user_error("property %s is not bool\n", name);
    return(property_bool(p));
}


/* Set the value of a bool property

   @param[in] name is the name of the property to set

   @param[in] b is the value to store in the property
*/
void set_bool_property(const char* name, bool b)
{
    property p = get_property(name, false);
    if (! property_bool_p(p))
	property_user_error("property %s is not bool\n", name);
    property_bool(p) = b;
}


/* Get the value of a string property

   @param[in] name is the name of the property to look-up

   @return the string value, not to be freed
*/
const char* get_string_property(const char* name)
{
    property p = get_property(name, false);
    if (! property_string_p(p))
	property_user_error("property %s is not string\n", name);
    return property_string(p);
}


/* Get the value of a string property or ask to the user if it does not
   has a non null string value

   @param[in] name is the name of the property to look-up

   @param[in] question is the sentence to display when asking for a value
   because the property has no valid string value

   @return the string value, not to be freed
*/
const char* get_string_property_or_ask(const char* name,const char question[])
{
    property p = get_property(name, false);
    if (! property_string_p(p))
        property_user_error("property %s is not string\n", name);
    string s = property_string(p);
    while(!s || string_undefined_p(s) || s[0] == '\0' )
        s = user_request(question);

    return s;
}


/* Set the value of a string property

   @param[in] name is the name of the property to set

   @param[in] s is the value to store in the property. This string is
   strdup()ed in this function
*/
void set_string_property(const char* name, const char* s)
{
    property p = get_property(name, false);
    if (! property_string_p(p))
	property_user_error("property %s is not string\n", name);
    free(property_string(p));
    property_string(p) = strdup(s);
}


/* Get the value of an integer property

   @param[in] name is the name of the property to look-up

   @return the integer value
*/
int
get_int_property(const char* name)
{
    property p = get_property(name, false);
    if (! property_int_p(p))
	property_user_error("property %s is not int\n", name);
    return property_int(p);
}


/* Set the value of an integer property

   @param[in] name is the name of the property to set

   @param[in] i is the value to store in the property
*/
void set_int_property(const char* name, int i)
{
    property p = get_property(name, false);
    if (! property_int_p(p))
	property_user_error("property %s is not int\n", name);
    property_int(p) = i;
}

void fprint_property_direct(FILE * fd, const char* pname)
{
    property p = get_property(pname, true);
    if (property_undefined_p(p))
	fprintf(fd, "# undefined property %s\n", pname);
    else
	switch (property_tag(p)) {
	case is_property_bool:
	    fprintf(fd, "%s", property_bool(p)? "TRUE": "FALSE");
	    break;
	case is_property_int:
	    fprintf(fd, "%td", property_int(p));
	    break;
	case is_property_string:
	    fprintf(fd, "%s", property_string(p));
	    break;
	default:
	    fprintf(fd, "# bad property type for %s\n", pname);
	}

}

void fprint_property(FILE * fd, const char* pname)
{
    property p = get_property(pname, true);
    if (property_undefined_p(p))
	fprintf(fd, "# undefined property %s\n", pname);
    else
	switch (property_tag(p)) {
	case is_property_bool:
	    fprintf(fd, "%s %s\n", pname, property_bool(p)? "TRUE": "FALSE");
	    break;
	case is_property_int:
	    fprintf(fd, "%s %td\n", pname, property_int(p));
	    break;
	case is_property_string:
	    fprintf(fd, "%s \"%s\"\n", pname, property_string(p));
	    break;
	default:
	    fprintf(fd, "# bad property type for %s\n", pname);
	}
}

void fprint_properties(FILE * fd)
{
  if (pl)
  { // dump all sorted properties in fd.
    gen_array_t array = gen_array_make(0);
    fprintf(fd, "# PIPS PROPERTIES\n");
    HASH_MAP(name, p, gen_array_dupappend(array, name), pl);
    gen_array_sort(array);
    GEN_ARRAY_FOREACH(string, name, array)
      fprint_property(fd, name);
    gen_array_full_free(array);
  }
  else fprintf(fd, "# NO PIPS PROPERTIES...\n");
}

/******************************************************* TOP-LEVEL INTERFACE */

static string get_property_file_name(void)
{
    string dir_name = db_get_meta_data_directory(),name;
    int stat = asprintf(&name,"%s/properties",dir_name);
    assert(stat != -1);
    free(dir_name); return name;
}

/* when opening a workspace, retrieve the properties.
 * @return whether okay
 */
bool open_properties(void)
{
    string file_name = get_property_file_name();
    FILE * file = check_fopen(file_name, "r");
    if (!file) return false;
    if (pl) hash_table_free(pl), pl = (hash_table) NULL; /* lazy clean... */
    parse_properties_file(file);
    safe_fclose(file, file_name);
    free(file_name);
    return true;
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

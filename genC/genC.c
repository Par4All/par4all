/*
 * $Id$
 *
 * Version which generates typed newgen structures.
 *
 * $Log: genC.c,v $
 * Revision 1.33  1998/04/11 12:23:42  coelho
 * make based on gen_alloc on second thought.
 * forgotten special case of tabulated fixed.
 * initial definitions fixed for import.
 *
 * Revision 1.32  1998/04/11 11:49:00  coelho
 * new version which generate struct.
 */

#include <stdio.h>
#include <ctype.h>
#include <string.h>

#include "genC.h"
#include "newgen_include.h"

#define INDENT "  "
#define FIELD  "field_"
#define STRUCT "_newgen_struct_"

#define IS_TAB(x) ((x)==Tabulated_bp)
#define TYPE(bp) (bp-Domains-Number_imports-Current_start)

/* GEN_SIZE returns the size (in gen_chunks) of an object of type defined by
 * the BP type.
 */
int gen_size(struct gen_binding * bp)
{
    int overhead = GEN_HEADER + IS_TABULATED(bp);

    switch (bp->domain->ba.type) 
    {
    case BASIS_DT:
    case ARRAY_DT:
    case LIST_DT:
    case SET_DT:
	return overhead + 1;
    case CONSTRUCTED_DT:
	if (bp->domain->co.op == OR_OP)
	    return overhead + 2;
	else if (bp->domain->co.op == AND_OP) 
	{
	    int size ;
	    struct domainlist * dlp = bp->domain->co.components ;
	    for( size=0 ; dlp != NULL ; dlp=dlp->cdr, size++ );
	    return overhead + size;
	}
	else if (bp->domain->co.op == ARROW_OP)
	    return overhead+1;
    default:
	fatal( "gen_size: Unknown type %s\n", itoa( bp->domain->ba.type )) ;
	return -1; /* to avoid a gcc warning */
	/*NOTREACHED*/
    }
}

/* returns s duplicated and case-uppered.
 */
static string strup(string s)
{
    string r = strdup(s), p=r;
    while (*p) *p=toupper(*p), p++;
    return r;
}

/* newgen type name for holder.
 * could/should always be gen_chunk?
 */
static string newgen_type_name(union domain * dp)
{
    switch (dp->ba.type) {
    case BASIS_DT: {
	struct gen_binding * bp = dp->ba.constructand;
	if (IS_INLINABLE(bp)) return "gen_chunk";
	else return bp->name;
    }
    case SET_DT: return "set";
    case LIST_DT: return "list";
    }
    return "";
}

static string newgen_argument_type_name(union domain * dp)
{
    switch (dp->ba.type) {
    case BASIS_DT: return dp->ba.constructand->name;
    case LIST_DT: return "list";
    case SET_DT: return "set";
    default: fatal("[newgen_argument_type] unexpected domain type %d\n", 
		   dp->ba.type);
    }    
    return NULL;
}

/* what to add to the field to access a given primitive type,
 * which was typically declared as a gen_chunk.
 * the field is the first char of the type name.
 */
static char newgen_access_name(union domain * dp)
{
    if (dp->ba.type==BASIS_DT && IS_INLINABLE(dp->ba.constructand))
	return dp->ba.constructand->name[0];
    else
	return '\0';
}

static string newgen_kind_label(union domain * dp)
{
    switch (dp->ba.type) {
    case SET_DT: return "{}";
    case LIST_DT: return "*";
    case ARRAY_DT: return "[]";
    default: return "";
    }
}

static void generate_make(
    FILE * header,
    FILE * code,
    struct gen_binding * bp,
    int domain_type,
    int operator)
{
    string name = bp->name;
    union domain * dom = bp->domain;
    struct domainlist * dlp;
    int i;

    /* HEADER
     */
    fprintf(header, "extern %s make_%s(", name, name);

    switch (domain_type)
    {
    case CONSTRUCTED_DT:
	switch (operator)
	{
	case AND_OP:
	    for (i=1, dlp=dom->co.components; dlp!=NULL; dlp=dlp->cdr, i++)
		fprintf(header, "%s%s", i==1? "": ", ",
			newgen_argument_type_name(dlp->domain));
	    break;
	case OR_OP:
	    fprintf(header, "int, void *");
	    break;
	case ARROW_OP:
	    fprintf(header, "void");
	    break;
	}
	break;
    case LIST_DT:
    case SET_DT:
    case ARRAY_DT:
	fprintf(header, "%s", newgen_argument_type_name(dom));
	break;
    default: 
	fatal("[generate_make] unexpected domain type tag %d\n", domain_type);
    }

    fprintf(header, ");\n");

    /* CODE
     */
    fprintf(code, "%s make_%s(", name, name);
    
    switch (domain_type)
    {
    case CONSTRUCTED_DT:
	switch (operator)
	{
	case AND_OP:
	    for (i=1, dlp=dom->co.components; dlp!=NULL; dlp=dlp->cdr, i++)
		fprintf(code, "%s%s a%d", i==1? "": ", ",
			newgen_argument_type_name(dlp->domain), i);
	    break;
	case OR_OP:
	    fprintf(code, "int tag, void * val");
	    break;
	case ARROW_OP:
	    fprintf(code, "void");
	    break;
	}
	break;
    case LIST_DT:
    case SET_DT:
    case ARRAY_DT:
	fprintf(code, "%s a", newgen_argument_type_name(dom));
	break;
    }

    fprintf(code,
	    ")\n{ return (%s) "
	    "gen_alloc(%d*sizeof(gen_chunk), GEN_CHECK_ALLOC, %s_domain",
	    name, gen_size(bp), name);
    
    
    switch (domain_type)
    {
    case CONSTRUCTED_DT:
	switch (operator)
	{
	case AND_OP:
	    for (i=1, dlp=dom->co.components; dlp!=NULL; dlp=dlp->cdr, i++)
		fprintf(code, ", a%d", i);
	    break;
	case OR_OP:
	    fprintf(code, ", tag, val");
	    break;
	case ARROW_OP:
	    break;
	}
	break;
    case LIST_DT:
    case SET_DT:
    case ARRAY_DT:
	fprintf(code, ", a");
	break;
    }

    fprintf(code, "); }\n");
}

static void generate_struct_members(
    FILE * out,  
    struct gen_binding * bp,
    int domain_type,
    int operator)
{
    union domain * dom = bp->domain;
    struct domainlist * dlp;
    string offset = "";
    int i;

    /* generate the structure
     */
    fprintf(out, 
	    "struct " STRUCT "%s {\n"
	    INDENT "gen_chunk type; /* int */\n",
	    bp->name);
 
    /* there is an additionnal field in tabulated domains.
     */
    if (IS_TABULATED(bp))
	fprintf(out, INDENT "gen_chunk index; /* int */\n");

    if (domain_type==CONSTRUCTED_DT && operator==OR_OP) {
	fprintf(out, INDENT "gen_chunk tag; /* int */\n" INDENT "union {\n");
	offset = INDENT;
    }

    if ((domain_type==CONSTRUCTED_DT && operator==ARROW_OP) ||
	domain_type==LIST_DT || 
	domain_type==ARRAY_DT || 
	domain_type==SET_DT)
	fprintf(out, INDENT "gen_chunk h; /* holder */\n");

    if (domain_type==CONSTRUCTED_DT && operator!=ARROW_OP)
	for (i=1, dlp=dom->co.components; dlp!=NULL; dlp=dlp->cdr, i++)
	    fprintf(out, "%s" INDENT "%s _%s_%s_" FIELD "; /* %s:%s%s */\n", 
		    offset,
		    newgen_type_name(dlp->domain),
		    bp->name, dlp->domain->ba.constructor,
		    dlp->domain->ba.constructor,
		    dlp->domain->ba.constructand->name,
		    newgen_kind_label(dlp->domain));

    if (domain_type==CONSTRUCTED_DT && operator==OR_OP) 
	fprintf(out, INDENT "} u;\n");
    
    fprintf(out, "};\n");
}

/* access to members are managed thru macros.
 * cannot be functions because assign would not be possible.
 * it would be better to avoid having field names that appear twice... 
 */
static void generate_access_members(
    FILE * out, 
    struct gen_binding * bp,
    int domain_type,
    int operator)
{
    union domain * dom = bp->domain;
    struct domainlist * dlp;
    string in_between, name=bp->name;
    int i;

    fprintf(out, "#define %s_domain_number(x) ((x)->type.i)\n", name);

    if (domain_type==CONSTRUCTED_DT && operator==OR_OP) {
	in_between = "u.";
	fprintf(out, "#define %s_tag(x) ((x)->tag.i)\n", name);
    }
    else in_between = "";
    
    if (domain_type==CONSTRUCTED_DT && operator==ARROW_OP) 
	fprintf(out, "#define %s_hash_table(x) ((x)->h.h)\n", name);

    if (domain_type==LIST_DT || domain_type==ARRAY_DT || domain_type==SET_DT)
	fprintf(out, "#define %s_%s(x) ((x)->h.%c)\n",
		name, dom->ba.constructor, 
		(domain_type==LIST_DT)? 'l':
		(domain_type==SET_DT)? 's': 'a');
    
    if (domain_type==CONSTRUCTED_DT && operator!=ARROW_OP)
	for (i=1, dlp=dom->co.components; dlp!=NULL; dlp=dlp->cdr, i++)
	{
	    char c;
	    if (operator==OR_OP)
		fprintf(out, 
			"#define is_%s_%s (%d)\n"
			"#define %s_%s_p(x) (%s_tag(x)==is_%s_%s)\n",
			name, dlp->domain->ba.constructor, i,
			name, dlp->domain->ba.constructor, 
			name, name, dlp->domain->ba.constructor);
	    fprintf(out, 
		    "#define %s_%s_(x) %s_%s(x)\n" /* hack compatible */
		    "#define %s_%s(x) ((x)->%s_%s_%s_" FIELD,
		    name, dlp->domain->ba.constructor, 
		    name, dlp->domain->ba.constructor, 
		    name, dlp->domain->ba.constructor, in_between, 
		    name, dlp->domain->ba.constructor);
	    c = newgen_access_name(dlp->domain);
	    if (c) fprintf(out, ".%c", c);
	    fprintf(out, ")\n");
	}
}

static void generate_constructed(
    FILE * header, 
    FILE * code,
    struct gen_binding * bp,
    int operator)
{
    generate_make(header, code, bp, CONSTRUCTED_DT, operator);
    generate_struct_members(header, bp, CONSTRUCTED_DT, operator);
    generate_access_members(header, bp, CONSTRUCTED_DT, operator);
}

static void generate_not_constructed(
    FILE * header,
    FILE * code,
    struct gen_binding * bp,
    int domain_type)
{
    generate_make(header, code, bp, domain_type, UNDEF_OP);
    generate_struct_members(header, bp, domain_type, UNDEF_OP);
    generate_access_members(header, bp, domain_type, UNDEF_OP);
}

/* newgen function (->) specific stuff
 */
static void generate_arrow(
    FILE * header,
    FILE * code,
    struct gen_binding * bp)
{
    union domain * dom, * key, * val;
    string name, kname, vname, Name;
    char kc, vc;

    dom = bp->domain;
    key = dom->co.components->domain;
    val = dom->co.components->cdr->domain;

    name = bp->name;
    Name = strup(name);
    kname = key->ba.constructand->name;
    vname = val->ba.constructand->name;

    /* how to extract the key and value from the hash_table chunks.
     */
    kc = newgen_access_name(key);
    if (kc=='\0') kc = 'p';
    vc = newgen_access_name(val);
    if (vc=='\0') vc = 'p';

    fprintf(header,
	    "#define %s_key_type %s\n"
	    "#define %s_value_type %s\n"
	    "#define %s_MAP(k,v,c,f) FUNCTION_MAP(%s,%c,%c,k,v,c,f)\n"
	    "extern %s apply_%s(%s, %s);\n"
	    "extern void update_%s(%s, %s, %s);\n"
	    "extern void extend_%s(%s, %s, %s);\n"
	    "extern void delete_%s(%s, %s);\n"
	    "extern bool bound_%s_p(%s, %s);\n",
	    name, kname, /* key type */
	    name, vname, /* val type */
	    Name, name, kc, vc, /* MAP */
	    vname, name, name, kname, /* apply */
	    name, name, kname, vname, /* update */
	    name, name, kname, vname, /* extend */
	    name, name, kname, /* delete */
	    name, name, kname /* bound_p */);

    fprintf(code, 
	    "%s apply_%s(%s f, %s k)\n"
	    "{ return (%s) HASH_GET(%c, %c, %s_hash_table(f), k); }\n"
	    "void update_%s(%s f, %s k, %s v)\n"
	    "{ HASH_UPDATE(%c, %c, %s_hash_table(f), k, v); }\n"
	    "void extend_%s(%s f, %s k, %s v)\n"
	    "{ HASH_EXTEND(%c, %c, %s_hash_table(f), k, v); }\n"
	    "void delete_%s(%s f, %s k)\n"
	    "{ HASH_DELETE(%c, %c, %s_hash_table(f), k); }\n"
	    "bool bound_%s_p(%s f, %s k)\n"
	    "{ return HASH_BOUND_P(%c, %c, %s_hash_table(f), k); }\n",
	    vname, name, name, kname, vname, kc, vc, name, /* apply */
	    name, name, kname, vname, kc, vc, name, /* update */
	    name, name, kname, vname, kc, vc, name, /* extend */
	    name, name, kname, kc, vc, name, /* delete */
	    name, name, kname, kc, vc, name /* bound_p */);

    free(Name);
}

/* generates a needed type declaration.
 */
static void
generate_safe_definition(
    FILE * out, 
    struct gen_binding * bp, 
    string file)
{
    string name = bp->name, Name = strup(name);
    int index = TYPE(bp);

    if (!IS_EXTERNAL(bp) && !IS_IMPORT(bp))
	fprintf(out, 
		"#define %s_domain (_gen_%s_start+%d)\n",
		name, file, index);

    fprintf(out, 
	   "#if !defined(_newgen_%s_domain_defined_)\n"
	   "#define _newgen_%s_domain_defined_\n",
	    name, name);

    if (IS_EXTERNAL(bp))
	/* kind of a bug here: if the very same externals appears
	 * several times, they are not sure to be attributed the same
	 * number in different files with different include orders.
	 * externals should really be global?
	 */
	fprintf(out, 
		"#define %s_NEWGEN_EXTERNAL (_gen_%s_start+%d)\n",
		Name, file, index);
    else
	fprintf(out, 
		"typedef struct " STRUCT "%s * %s;\n", 
		name, name);
    
    fprintf(out, "#endif /* _newgen_%s_domain_defined_ */\n\n", name);
    free(Name);
}

/* generate the needed stuff for bp. 
 */
static void 
generate_domain(
    FILE * header,
    FILE * code,
    struct gen_binding * bp)
{
    union domain * dp = bp->domain;
    string name = bp->name, Name = strup(bp->name);
    
    if (!IS_EXTERNAL(bp))
    {
	/* assumes a preceeding safe definition.
	 * non specific (and/or...) stuff. 
	 */
	fprintf(header, 
		"/* %s\n */\n"
		"#define %s(x) ((%s)(x))\n"
		"#define %s_TYPE %s\n"
		"#define %s_undefined ((%s)gen_chunk_undefined)\n"
		"#define %s_undefined_p(x) ((x)==%s_undefined)\n"
		"\n"
		"extern %s copy_%s(%s);\n"
		"extern void write_%s(FILE *, %s);\n"
		"extern %s read_%s(FILE *);\n"
		"extern void free_%s(%s);\n"
		"extern %s check_%s(%s);\n"
		"extern bool %s_consistent_p(%s);\n",
		Name,
		Name, name, /* defines... */
		Name, name,
		name, name,
		name, name,
		name, name, name, /* copy */
		name, name, /* write */
		name, name, /* read */
		name, name, /* free */
		name, name, name, /* check */  
		name, name  /* consistent */);
	
	fprintf(code,
		"/* %s\n */\n"
		"%s copy_%s(%s p)\n"
		"{ return (%s) gen_copy_tree((gen_chunk*)p); }\n"
		"void write_%s(FILE * f, %s p)\n"
		"{ gen_write(f,(gen_chunk*)p); }\n"
		"%s read_%s(FILE * f)\n"
		"{ return (%s) gen_read(f); }\n"
		"void free_%s(%s p)\n"
		"{ gen_free((gen_chunk*)p); }\n"
		"%s check_%s(%s p)\n"
		"{ return (%s) gen_check((gen_chunk*)p, %s_domain); }\n"
		"bool %s_consistent_p(%s p)\n"
		"{ check_%s(p); return gen_consistent_p((gen_chunk*)p); }\n",
		Name,
		name, name, name, name, /* copy */
		name, name, /* write */
		name, name, name, /* read */
		name, name, /* free */
		name, name, name, name, name, /* check */
		name, name, name /* consistent */);
    }
    
    switch (dp->ba.type) {
    case CONSTRUCTED_DT:
	switch (dp->co.op) {
	case AND_OP: 
	    generate_constructed(header, code, bp, AND_OP);
	    break;
	case OR_OP:
	    generate_constructed(header, code, bp, OR_OP);
	    break;
	case ARROW_OP: 
	    generate_constructed(header, code, bp, ARROW_OP);
	    generate_arrow(header, code, bp);
	    break;	    
	default:
	    fatal("[generate_domain] unexpected constructed %d\n", dp->co.op);
	}
	break;
    case LIST_DT:
	generate_not_constructed(header, code, bp, LIST_DT);
	break;
    case SET_DT:
	generate_not_constructed(header, code, bp, SET_DT);
	break;
    case ARRAY_DT:
	generate_not_constructed(header, code, bp, ARRAY_DT);
	break;
    case EXTERNAL_DT:
	/* nothing to generate at the time. */
	break;
    default:
	fatal("[generate_domain] unexpected domain type %d\n", dp->ba.type);
    }

    fprintf(header, "\n");
    fprintf(code, "\n");

    free(Name);
}

static FILE * fopen_suffix(string prefix, string suffix)
{
    FILE * f;
    string r = (string) malloc(strlen(prefix)+strlen(suffix)+1);
    if (r==NULL) fatal("[fopen_suffix] no more memory\n");
    strcpy(r, prefix);
    strcat(r, suffix);
    f = fopen(r, "w");
    if (f==NULL) fatal("[fopen_suffix] of %s failed\n", r);
    free(r);
    return f;
}

#define DONT_TOUCH						\
  "/*\n"							\
  " * THIS FILE HAS BEEN AUTOMATICALLY GENERATED BY NEWGEN.\n"	\
  " *\n"							\
  " * PLEASE DO NOT MODIFY IT.\n"				\
  " */\n"

/* generate the code necessary to manipulate every internal
 * non-inlinable type in the Domains table.
 */
void gencode(string file)
{
    int i;
    FILE * header, * code;

    if (file==NULL) fatal("[gencode] no file name specified\n");

    /* header = fopen_suffix(file, ".h"); */
    header = stdout;
    code = fopen_suffix(file, ".c");

    fprintf(header, DONT_TOUCH);

    fprintf(code, DONT_TOUCH);
    fprintf(code, 
	    "\n"
	    "#include <stdio.h>\n"
	    "#include <stdlib.h>\n"
	    "#include \"genC.h\"\n"
	    "#include \"%s.h\"\n",
	    file);

    for (i=0; i<MAX_DOMAIN; i++)
    {
	struct gen_binding * bp = &Domains[i];
	if (bp->name && !IS_INLINABLE(bp) && !IS_TAB(bp))
	    generate_safe_definition(header, bp, file);
    }

    for (i=0; i<MAX_DOMAIN; i++)
    {
	struct gen_binding * bp = &Domains[i];
	if (bp->name && !IS_INLINABLE(bp) && !IS_IMPORT(bp) && !IS_TAB(bp))
	    generate_domain(header, code, bp);
    }

    /* fclose(header); */
    fclose(code);
}

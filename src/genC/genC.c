/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

  This file is part of NewGen.

  NewGen is free software: you can redistribute it and/or modify it under the
  terms of the GNU General Public License as published by the Free Software
  Foundation, either version 3 of the License, or any later version.

  NewGen is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
  License for more details.

  You should have received a copy of the GNU General Public License along with
  NewGen.  If not, see <http://www.gnu.org/licenses/>.

*/

/*
  generates typed newgen structures.
*/
#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>

#include "genC.h"
#include "newgen_include.h"

#undef gen_recurse
#undef gen_context_recurse

#define FIELD  "" /* was: "field_" */
#define STRUCT "_newgen_struct_" /* structure names */

#define OPTIMIZE_NEWGEN "OPTIMIZE_NEWGEN"

#define IS_TAB(x) ((x)==Tabulated_bp)

/* non user domain must be taken care from outside? */
/* #define FIRST_USER_DOMAIN (7) */
#define TYPE(bp) (bp-Domains-Number_imports-Current_start)

#define DomainNumberError \
  "\"[newgen internal error]\""  \
  "\"inconsistent domain number for %s: %%d (expecting %%d)\\n\""

/* Might be used to generate optimized versions
   (say macros instead of functions).
 */
static int sharp_ifopt(FILE * out)
{ return fprintf(out, "#if defined(" OPTIMIZE_NEWGEN ")\n"); }

static int sharp_else(FILE * out)
{ return fprintf(out, "#else\n"); }

static int sharp_endif(FILE * out)
{ return fprintf(out, "#endif /* " OPTIMIZE_NEWGEN " */\n"); }

/* GEN_SIZE returns the size (in gen_chunks) of an object of type defined by
 * the BP type.
 */
int gen_size(int domain)
{
  struct gen_binding * bp = Domains+domain;
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

#define same_size(t) (sizeof(t)==sizeof(gen_chunk))

static bool inline_directly(union domain * dp)
{
  if (dp->ba.type==BASIS_DT)
  {
    struct gen_binding * bp = dp->ba.constructand;
    string name = bp->name;
    if (!IS_INLINABLE(bp)) return false;
    if ((same_string_p(name, "int") && same_size(intptr_t)) ||
	(same_string_p(name, "string") && same_size(string)) ||
	(same_string_p(name, "unit") && same_size(unit)))
      return true;
  }
  return false;
}

/* bof... */
static string int_type(void)
{
  return same_size(intptr_t) ? "intptr_t": "gen_chunk";
}

static string int_type_access_complement(void)
{
  return same_size(intptr_t) ? "": ".i";
}

/* newgen type name for holder.
 * could/should always be gen_chunk?
 */
static string newgen_type_name(union domain * dp)
{
  switch (dp->ba.type) {
  case BASIS_DT: {
    struct gen_binding * bp = dp->ba.constructand;
    if (IS_INLINABLE(bp))
      if (!inline_directly(dp))
	return "gen_chunk";
    if (same_string_p(bp->name, "int"))
      /* Even if the NewGen name is int, the C name can be intptr_t: */
      return int_type();
    return bp->name;
  }
  case SET_DT: return "set";
  case LIST_DT: return "list";
  case ARRAY_DT: return dp->ar.element->name;
  case CONSTRUCTED_DT:
    switch (dp->co.op) {
    case ARROW_OP: return "hash_table";
    default:
      break;
    }
  default: fatal("[newgen_type_name] unexpected domain type %d\n",
		 dp->ba.type);
  }
  return NULL;
}

/* C type name for generated function arguments.
 */
static string newgen_argument_type_name(union domain * dp)
{
  switch (dp->ba.type) {
  case BASIS_DT:
    if (same_string_p(dp->ba.constructand->name, "int"))
      /* Even if the NewGen name is int, the C name can be intptr_t: */
      return int_type();
    return dp->ba.constructand->name;
  case LIST_DT: return "list";
  case SET_DT: return "set";
  case ARRAY_DT: return dp->ar.element->name;
  default: fatal("[newgen_argument_type_name] unexpected domain type %d\n",
		 dp->ba.type);
  }
  return NULL;
}

static string newgen_type_name_close(union domain * dp)
{
  if (dp->ba.type==ARRAY_DT) return " *"; /* pointer */
  return "";
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

/* just to generate comprehensive comments.
 */
static string newgen_kind_label(union domain * dp)
{
  switch (dp->ba.type) {
  case SET_DT: return "{}";
  case LIST_DT: return "*";
  case ARRAY_DT: return "[]";
  default: return "";
  }
}

/* make is bigger, thus I put it in a separate function.
 */
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
  int domain = bp-Domains;
  int i;

  /* HEADER
   */
  fprintf(header, "extern %s make_%s(", name, name);

  switch (domain_type) {
  case CONSTRUCTED_DT:
    switch (operator) {
    case AND_OP:
      for (i=1, dlp=dom->co.components; dlp!=NULL; dlp=dlp->cdr, i++)
	fprintf(header, "%s%s%s", i==1? "": ", ",
		newgen_argument_type_name(dlp->domain),
		newgen_type_name_close(dlp->domain));
      break;
    case OR_OP:
      fprintf(header, "enum %s_utype, void *", name);
      break;
    case ARROW_OP:
      fprintf(header, "void");
      break;
    }
    break;
  case LIST_DT:
  case SET_DT:
  case ARRAY_DT:
    fprintf(header, "%s%s",
	    newgen_argument_type_name(dom),
	    newgen_type_name_close(dom));
    break;
  default:
    fatal("[generate_make] unexpected domain type tag %d\n", domain_type);
  }

  fprintf(header, ");\n");

  /* CODE
   */
  fprintf(code, "%s make_%s(", name, name);
  switch (domain_type) {
  case CONSTRUCTED_DT:
    switch (operator) {
    case AND_OP:
      for (i=1, dlp=dom->co.components; dlp!=NULL; dlp=dlp->cdr, i++)
	fprintf(code, "%s%s%s a%d", i==1? "": ", ",
		newgen_argument_type_name(dlp->domain),
		newgen_type_name_close(dlp->domain), i);
      break;
    case OR_OP:
      fprintf(code, "enum %s_utype tag, void * val", name);
      break;
    case ARROW_OP:
      fprintf(code, "void");
      break;
    }
    break;
  case LIST_DT:
  case SET_DT:
  case ARRAY_DT:
    fprintf(code, "%s%s a",
	    newgen_argument_type_name(dom),
	    newgen_type_name_close(dom));
    break;
  }

  fprintf(code,
	  ") {\n"
	  "  return (%s) "
	  "gen_alloc(%d*sizeof(gen_chunk), GEN_CHECK_ALLOC, %s_domain",
	  name, gen_size(domain), name);
  switch (domain_type) {
  case CONSTRUCTED_DT:
    switch (operator) {
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
  fprintf(code, ");\n}\n");

  /* additionnal constructors for OR,
     so as to improve type checking.
  */
  if (domain_type==CONSTRUCTED_DT && operator==OR_OP) {
    for (dlp=dom->co.components; dlp!=NULL; dlp=dlp->cdr) {
      /* check for unit... */
      string field = dlp->domain->ba.constructor;
      string typen = newgen_argument_type_name(dlp->domain);
      if(!strcmp(typen, UNIT_TYPE_NAME)) {
	/* UNIT case */
	/* header */
	fprintf(header,
		"extern %s make_%s_%s(void);\n",
		name, name, field);
	/* code */
	fprintf(code,
		"%s make_%s_%s(void) {\n"
		"  return make_%s(is_%s_%s, UU);\n"
		"}\n",
		name, name, field, name, name, field);
      }
      else {
	/* header */
	fprintf(header, "extern %s make_%s_%s(%s);\n",
		name, name, field, typen);
	/* code */
	fprintf(code,
		"%s make_%s_%s(%s _field_) {\n"
		"  return make_%s(is_%s_%s, (void*)(intptr_t) _field_);\n"
		"}\n",
		name, name, field, typen,
		name, name, field);
      }
    }
  }
}

/* generate the struct for bp.
 */
static void generate_struct_members(
  FILE * out,
  struct gen_binding * bp,
  int domain_type,
  int operator)
{
  union domain * dom = bp->domain;
  struct domainlist * dlp;
  string offset = "";

  /* generate the structure
   */
  fprintf(out,
	  "struct " STRUCT "%s_ {\n"
	  "  %s _type_;\n",
	  bp->name, int_type());

  /* there is an additionnal field in tabulated domains.
   */
  if (IS_TABULATED(bp))
    fprintf(out,
	    "  %s _%s_index__;\n",
	    int_type(), bp->name);

  if (domain_type==CONSTRUCTED_DT && operator==OR_OP) {
    fprintf(out,
	    "  enum %s_utype _%s_tag__;\n"
	    "  union {\n",
	    bp->name, bp->name);
    offset = "  ";
  }

  if ((domain_type==CONSTRUCTED_DT && operator==ARROW_OP) ||
      domain_type==LIST_DT ||
      domain_type==SET_DT)
    fprintf(out,
	    "  %s _%s_holder_;\n",
	    newgen_type_name(dom), bp->name);

  if (domain_type==CONSTRUCTED_DT && operator!=ARROW_OP) {
    /* generate struct fields */
    for (dlp=dom->co.components; dlp!=NULL; dlp=dlp->cdr)
      fprintf(out, "%s  %s%s _%s_%s_" FIELD "; /* %s:%s%s */\n",
	      offset,
	      newgen_type_name(dlp->domain),
	      newgen_type_name_close(dlp->domain),
	      bp->name, dlp->domain->ba.constructor,
	      dlp->domain->ba.constructor,
	      dlp->domain->ba.constructand->name,
	      newgen_kind_label(dlp->domain));
  }

  if (domain_type==CONSTRUCTED_DT && operator==OR_OP)
    fprintf(out, "  } _%s_union_;\n", bp->name);

  fprintf(out, "};\n\n");
}

static void generate_union_type_descriptor(
  FILE * out,
  struct gen_binding * bp,
  int domain_type,
  int operator)
{
  union domain * dom = bp->domain;
  struct domainlist * dlp;
  string name=bp->name;

  if (domain_type==CONSTRUCTED_DT &&
      operator==OR_OP && dom->co.components!=NULL)
  {
    fprintf(out,"enum %s_utype {\n", name);
    for (dlp=dom->co.components; dlp!=NULL; dlp=dlp->cdr) {
      string field = dlp->domain->ba.constructor;
      fprintf(out,
	      "  is_%s_%s%s\n",
	      name, field, (dlp->cdr == NULL)? "": ",");
    }
    fprintf(out,"};\n");
  }
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
  bool in_between;
  string name=bp->name;

  fprintf(out,
	  "#define %s_domain_number(x) ((x)->_type_%s)\n",
	  name, int_type_access_complement());

  if (domain_type==CONSTRUCTED_DT && operator==OR_OP) {
    in_between = true;
    fprintf(out,
	    "#define %s_tag(x) ((x)->_%s_tag__%s)\n",
	    name, name, int_type_access_complement());
  }
  else in_between = false;

  if (domain_type==CONSTRUCTED_DT && operator==ARROW_OP)
    fprintf(out, "#define %s_hash_table(x) ((x)->_%s_holder_)\n",
	    name, name);

  if (domain_type==LIST_DT || domain_type==SET_DT)
    fprintf(out, "#define %s_%s(x) ((x)->_%s_holder_)\n",
	    name, dom->ba.constructor, name);

  if (domain_type==ARRAY_DT)
    fprintf(out, "#define %s_%s(x) ((x)->_%s_%s_" FIELD "\n",
	    name, dom->ba.constructor,
	    name, dom->ba.constructor);

  if (domain_type==CONSTRUCTED_DT && operator!=ARROW_OP) {

      /* accesses... */
      for (dlp=dom->co.components; dlp!=NULL; dlp=dlp->cdr) {
          char c;
          if(operator==OR_OP) {
              string field = dlp->domain->ba.constructor;
              fprintf(out,
                      "#define %s_%s_p(x) (%s_tag(x)==is_%s_%s)\n",
                      name, field, name, name, field);
          }
          fprintf(out,
                  "#define %s_%s_(x) %s_%s(x) /* old hack compatible */\n"
                  "#define %s_%s(x) ((x)->",
                  name, dlp->domain->ba.constructor,
                  name, dlp->domain->ba.constructor,
                  name, dlp->domain->ba.constructor);
          if (in_between) fprintf(out, "_%s_union_.", name);
          fprintf(out, "_%s_%s_" FIELD, name, dlp->domain->ba.constructor);
          c = newgen_access_name(dlp->domain);
          if (c && !inline_directly(dlp->domain))
              fprintf(out, ".%c", c);
          fprintf(out, ")\n");
      }
  }
}

/* constructed types: + x (and ->...)
 */
static void generate_constructed(
  FILE * header,
  FILE * code,
  struct gen_binding * bp,
  int operator)
{
  generate_union_type_descriptor(header, bp, CONSTRUCTED_DT, operator);
  generate_make(header, code, bp, CONSTRUCTED_DT, operator);
  fprintf(header, "\n");
  generate_struct_members(header, bp, CONSTRUCTED_DT, operator);
  generate_access_members(header, bp, CONSTRUCTED_DT, operator);
}

/* other types (direct * {} [])
 */
static void generate_not_constructed(
  FILE * header,
  FILE * code,
  struct gen_binding * bp,
  int domain_type)
{
  generate_make(header, code, bp, domain_type, UNDEF_OP);
  fprintf(header, "\n");
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
  // To deal with long int version:
  kname = newgen_argument_type_name(key);
  vname = newgen_argument_type_name(val);

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
	  "extern %s delete_%s(%s, %s);\n"
	  "extern bool bound_%s_p(%s, %s);\n",
	  name, kname, /* key type */
	  name, vname, /* val type */
	  Name, name, kc, vc, /* MAP */
	  vname, name, name, kname, /* apply */
	  name, name, kname, vname, /* update */
	  name, name, kname, vname, /* extend */
	  vname, name, name, kname, /* delete */
	  name, name, kname /* bound_p */);

  fprintf(code,
	  "%s apply_%s(%s f, %s k) {\n"
	  "  return (%s) (intptr_t)HASH_GET(%c, %c, %s_hash_table(f), k);\n"
	  "}\n"
	  "void update_%s(%s f, %s k, %s v) {\n"
	  "  HASH_UPDATE(%c, %c, %s_hash_table(f), k, (intptr_t)v);\n"
	  "}\n"
	  "void extend_%s(%s f, %s k, %s v) {\n"
	  "  HASH_EXTEND(%c, %c, %s_hash_table(f), k, (intptr_t)v);\n"
	  "}\n"
	  "%s delete_%s(%s f, %s k) {\n"
	  "  return (%s)(intptr_t) HASH_DELETE(%c, %c, %s_hash_table(f), k);\n"
	  "}\n"
	  "bool bound_%s_p(%s f, %s k) {\n"
	  "  return (intptr_t)HASH_BOUND_P(%c, %c, %s_hash_table(f), k);\n"
	  "}\n",
	  vname, name, name, kname, vname, kc, vc, name, /* apply */
	  name, name, kname, vname, kc, vc, name, /* update */
	  name, name, kname, vname, kc, vc, name, /* extend */
	  vname, name, name, kname, vname, kc, vc, name, /* delete */
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
	    "#define newgen_%s(p) (p) /* old hack compatible */\n"
	    "#define %s_NEWGEN_EXTERNAL (_gen_%s_start+%d)\n"
	    "#define %s_NEWGEN_DOMAIN (%s_NEWGEN_EXTERNAL)\n"
	    "#define %s_NEWGEN_DOMAIN (%s_NEWGEN_EXTERNAL)\n",
	    name,
	    Name, file, index,
	    Name, Name,
	    name, Name);
  else
    /* should not run if IS_IMPORT(bp)??? */
    fprintf(out,
	    "#define %s_NEWGEN_DOMAIN (%s_domain)\n"
	    "#define %s_NEWGEN_DOMAIN (%s_domain)\n"
	    "typedef struct " STRUCT "%s_ * %s;\n",
	    Name, name,
	    name, name,
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

  if (!IS_EXTERNAL(bp)) {
    /* assumes a preceeding safe definition.
     * non specific (and/or...) stuff.
     */
    fprintf(header,
            "/* %s\n */\n"
            "#define %s(x) ((%s)((x).p))\n"
            // foo_CAST FOO_CAST
            "#define %s_CAST(x) %s(x)\n"
            "#define %s_CAST(x) %s(x)\n"
            "#define %s_(x) ((x).e)\n"
            // foo_TYPE FOO_TYPE
            "#define %s_TYPE %s\n"
            "#define %s_TYPE %s\n"
            "#define %s_undefined ((%s)gen_chunk_undefined)\n"
            "#define %s_undefined_p(x) ((x)==%s_undefined)\n"
            /* what about assignment with checks?
            // something like:
            // #define FOO_assign(r,v) \
            // { FOO * _p = &(r), _v = (v); \
            //   FOO_check(r); FOO_check(v); \
            //   *_p = _v; \
            // }
            */
            "\n"
            "extern %s copy_%s(%s);\n"
            "extern void free_%s(%s);\n"
            "extern %s check_%s(%s);\n"
            "extern bool %s_consistent_p(%s);\n"
            "extern bool %s_defined_p(%s);\n"
            "#define gen_%s_cons gen_%s_cons\n"
            "extern list gen_%s_cons(%s, list);\n"
            "extern void %s_assign_contents(%s, %s);\n"
            "extern void %s_non_recursive_free(%s);\n",
            Name, // comments
            Name, name, // defines...
            name, Name,
            Name, Name,
            Name,
            Name, name, // XXX_TYPE
            name, name, // xxx_TYPE
            name, name,
            name, name,
            name, name, name, // copy
            name, name, // free
            name, name, name, // check
            name, name, // consistent
            name, name, // defined
            Name, name, // gen cons
            name, name,
            name, name, name, // assign contents
            name, name // non recursive free
      );

    fprintf(code,
            "/* %s\n */\n"
            "%s copy_%s(%s p) {\n"
            "  return (%s) gen_copy_tree((gen_chunk*) p);\n"
            "}\n"
            "void free_%s(%s p) {\n"
            "  gen_free((gen_chunk*) p);\n"
            "}\n"
            "%s check_%s(%s p) {\n"
            "  return (%s) gen_check((gen_chunk*) p, %s_domain);\n"
            "}\n"
            "bool %s_consistent_p(%s p) {\n"
            "  check_%s(p);\n"
            "  return gen_consistent_p((gen_chunk*) p);\n"
            "}\n"
            "bool %s_defined_p(%s p) {\n"
            "  return gen_defined_p((gen_chunk*) p);\n"
            "}\n"
            "list gen_%s_cons(%s p, list l) {\n"
            "  return gen_typed_cons(%s_NEWGEN_DOMAIN, p, l);\n"
            "}\n"
            "void %s_assign_contents(%s r, %s v) {\n"
            "  check_%s(r);\n"
            "  check_%s(v);\n"
            "  message_assert(\"defined references to domain %s\",\n"
            "                 %s_defined_p(r) && %s_defined_p(v));\n"
            "       memcpy(r, v, sizeof(struct " STRUCT "%s_));\n"
            "}\n"
            "void %s_non_recursive_free(%s p) {\n"
            "  // should clear up contents...\n"
            "  free(p);\n"
            "}\n",
            Name, // comments
            name, name, name, name, // copy
            name, name, // free
            name, name, name, name, name, // check
            name, name, name, // consistent
            name, name, // consistent
            name, name, Name, // gen cons
            name, name, name, // assign contents
            name, name, name, name, name, name,
            name, name // non recursive free
      );

    if (IS_TABULATED(bp)) {
      /* tabulated */
      fprintf(header,
	      "extern %s gen_find_%s(char *);\n"
	      "extern void write_tabulated_%s(FILE *);\n"
	      "extern void read_tabulated_%s(FILE *);\n",
	      name, name, /* find */
	      name, /* write */
	      name /* read */);
      fprintf(code,
	      "%s gen_find_%s(char* s) {\n"
	      "  return (%s) gen_find_tabulated(s, %s_domain);\n"
	      "}\n"
	      "void write_tabulated_%s(FILE* f) {\n"
	      "  (void) gen_write_tabulated(f, %s_domain);\n"
	      "}\n"
	      "void read_tabulated_%s(FILE* f) {\n"
	      "  int domain = gen_read_tabulated(f, 0);\n"
	      "  if (domain!=%s_domain) {\n"
	      "    fprintf(stderr, " DomainNumberError ",\n"
	      "            domain, %s_domain);\n"
	      "    abort();\n"
	      "  }\n"
	      "}\n",
	      name, name, name, name, /* find */
	      name, name, /* write */
	      name, name, name, name /* read */);
    }
    else {
      /* NOT tabulated */
      fprintf(header,
	      "extern void write_%s(FILE*, %s);\n"
	      "extern %s read_%s(FILE*);\n",
	      name, name, /* write */
	      name, name /* read */);
      fprintf(code,
	      "void write_%s(FILE* f, %s p) {\n"
	      "  gen_write(f, (gen_chunk*) p);\n"
	      "}\n"
	      "%s read_%s(FILE* f) {\n"
	      "  return (%s) gen_read(f);\n"
	      "}\n",
	      name, name, /* write */
	      name, name, name /* read */);
    }
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

/* fopen prefix + suffix.
 */
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

#define DONT_TOUCH                                              \
  "/*\n"                                                        \
  " * THIS FILE HAS BEEN AUTOMATICALLY GENERATED BY NEWGEN.\n"  \
  " *\n"                                                        \
  " * PLEASE DO NOT MODIFY IT.\n"                               \
  " */\n\n"

/* generate the code necessary to manipulate every internal
 * non-inlinable type in the Domains table.
 */
void gencode(string file)
{
  // Just a hack to pretend we use these functions
  void * no_warning = &sharp_ifopt - &sharp_else + &sharp_endif;
  int i;
  FILE * header, * code;

  if (file==NULL)
    fatal("[gencode] no file name specified (%p)\n", no_warning);

  if (sizeof(void *)!=sizeof(gen_chunk))
    fatal("[gencode] newgen fundamental layout hypothesis broken\n");

  /* header = fopen_suffix(file, ".h"); */
  header = stdout;
  code = fopen_suffix(file, ".c");

  fprintf(header, DONT_TOUCH);
  fprintf(code, DONT_TOUCH);

  for (i=0; i<MAX_DOMAIN; i++) {
    struct gen_binding * bp = &Domains[i];
    if (bp->name && !IS_INLINABLE(bp) && !IS_TAB(bp))
      if (IS_EXTERNAL(bp))
	fprintf(code, "typedef void * %s;\n", bp->name);
  }

  fprintf(code,
          "\n"
          "#include <stdio.h>\n"
          "#include <stdlib.h>\n"
          "#include <string.h>\n"
          "#include \"genC.h\"\n"
          "#include \"%s.h\"\n"
          "\n",
          file);

  /* first generate protected forward declarations.
   */
  for (i=0; i<MAX_DOMAIN; i++) {
    struct gen_binding * bp = &Domains[i];
    if (bp->name && !IS_INLINABLE(bp) && !IS_TAB(bp))
      generate_safe_definition(header, bp, file);
  }

  /* then generate actual declarations.
   */
  for (i=0; i<MAX_DOMAIN; i++) {
    struct gen_binding * bp = &Domains[i];
    if (bp->name && !IS_INLINABLE(bp) && !IS_IMPORT(bp) && !IS_TAB(bp))
      generate_domain(header, code, bp);
  }

  /* fclose(header); */
  fclose(code);
}

/*

 $Id$

 Copyright 1989-2014 MINES ParisTech
 Copyright 2009-2010 HPC-Project

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

#include "gfc2pips-private.h"

#include "c_parser_private.h"
#include "misc.h"
#include "text-util.h"
#include <stdio.h>

list gfc_module_callees = NULL;
list gfc2pips_list_of_declared_code = NULL;
list gfc2pips_list_of_loops = NULL;



/**
 * @brief generate an union of unique elements taken from A and B
 */
list gen_union(list a, list b) {
  list c = NULL;
  while(a) {
    if(!gen_in_list_p(CHUNK( CAR( a ) ), c))
      c = gen_cons(CHUNK( CAR( a ) ), c);
    POP( a );
  }
  while(b) {
    if(!gen_in_list_p(CHUNK( CAR( b ) ), c))
      c = gen_cons(CHUNK( CAR( b ) ), c);
    POP( b );
  }
  return c;
}


/**
 * @brief Add an entity to the list of callees
 *
 */
void gfc2pips_add_to_callees(entity e) {

  if(!intrinsic_entity_p(e) && strcmp_(entity_local_name(e), CurrentPackage)
      != 0) {
    gfc2pips_debug(5, "Add callee : %s\n", entity_local_name( e ) );
    gfc_module_callees = CONS(string,entity_local_name(e),gfc_module_callees);
  }
}

/**
 * Here are few utility functions to handle splitted files and do comparisons
 * with case insensitive
 */
/**
 * @brief replace lower case char by upper case ones
 */
char * str2upper(char s[]) {

  // FIXME Disabled !!!
  //  return s;

  int n = 0;
  if(s && s[n] != '\0') {
    do {
      s[n] = toupper(s[n]);
      n++;
    } while(s[n] != '\0');
  }
  return s;
}
/**
 * @brief replace lower case char by upper case ones
 */
char * strn2upper(char s[], size_t n) {
  while(n) {
    s[n - 1] = toupper(s[n - 1]);
    n--;
  }
  return s;
}
/**
 * @brief copy a string from the last char (to allow copy on itself)
 */
char * strrcpy(char *dest, __const char *src) {
  int i = strlen(src);
  while(i--)
    dest[i] = src[i];
  return dest;
}
/**
 * @brief insensitive case comparison
 */
int strcmp_(__const char *__s1, __const char *__s2) {
  char *a = str2upper(strdup(__s1));
  char *b = str2upper(strdup(__s2));
  int ret = strcmp(a, b);
  free(a);
  free(b);
  return ret;
}

/**
 * @brief insensitive case n-comparison
 */
int strncmp_(__const char *__s1, __const char *__s2, size_t __n) {
  char *a = str2upper(strdup(__s1));
  char *b = str2upper(strdup(__s2));
  int ret = strncmp(a, b, __n);
  free(a);
  free(b);
  return ret;
}

/**
 * @brief copy the file called *old to the file called *new
 */
int fcopy(const char* old, const char* new) {
  if(!old || !new)
    return 0;
  FILE * o = fopen(old, "r");
  if(o) {
    FILE * n = fopen(new, "w");
    if(n) {
      int c = fgetc(o);
      while(c != EOF) {
        fputc(c, n);
        c = fgetc(o);
      }
      fclose(n);
      fclose(o);
      return 1;
    }
    fclose(o);
    return 0;
  }
  return 0;
}

/**
 * @brief expurgates a string representing a REAL, could be a pre-prettyprinter
 * processing
 *
 * 1.0000000000e+00  becomes  1.
 * 1234.5670000e+18  becomes  1234.567e+18
 */
void gfc2pips_truncate_useless_zeroes(char *s) {
  char *start = s;
  bool has_dot = false;
  char *end_sci = NULL;//scientific output ?
  while(*s) {
    if(*s == '.') {
      has_dot = true;
      gfc2pips_debug(9,"found [dot] at %lu\n",s-start);
      s++;
      while(*s) {
        if(*s == 'e') {
          end_sci = s;
          break;
        }
        s++;
      }
      break;
    }
    s++;
  }
  if(has_dot) {
    int nb = 0;
    if(end_sci) {
      s = end_sci - 1;
    } else {
      s = start + strlen(start);
    }

    while(s > start) {
      if(*s == '0') {
        *s = '\0';
        nb++;
      } else {
        break;
      }
      s--;
    }
    gfc2pips_debug(9,"%d zero(s) retrieved\n", nb);
    /*if(*s=='.'){
     *s='\0';
     s--;
     gfc2pips_debug(9,"final dot retrieved\n");
     }*/
    if(end_sci) {
      if(strcmp(end_sci, "e+00") == 0) {
        *(s + 1) = '\0';
      } else if(s != end_sci - 1) {
        strcpy(s + 1, end_sci);
      }
    }
  }
}

void load_entities() {
  FILE *entities =
      (FILE *)safe_fopen((char *)gfc_option.gfc2pips_entities, "r");
  int read = gen_read_tabulated(entities, FALSE);
  safe_fclose(entities, (char *)gfc_option.gfc2pips_entities);
  pips_assert("entities were read", read==entity_domain);
}

void save_entities() {
  FILE *entities =
      (FILE *)safe_fopen((char *)gfc_option.gfc2pips_entities, "w");
  gen_write_tabulated(entities, entity_domain);
  safe_fclose(entities, (char *)gfc_option.gfc2pips_entities);
}

void pips_init() {

  static int initialized = FALSE;

  if(!initialized) {
    /* get NewGen data type description */
    //  gen_read_spec(ALL_SPECS);
    gen_read_spec(ri_spec,
                  text_spec,
                  c_parser_private_spec,
                  parser_private_spec,
                  (char*)NULL);

    gen_init_external(PVECTEUR_NEWGEN_EXTERNAL,
                      (void* (*)())vect_gen_read,
                      (void(*)())vect_gen_write,
                      (void(*)())vect_gen_free,
                      (void* (*)())vect_gen_copy_tree,
                      (int(*)())vect_gen_allocated_memory);

    // Pips init
    load_entities();

    initialized = TRUE;
  }
}

/**
 * void gfc2pips_get_use_st( void );
 * @brief This function is called by the GFC parser when encountering a USE
 * statement. It'll produce an entry in "ns2use" hashtable
 *
 */
hash_table ns2use = NULL;
void gfc2pips_get_use_st(void) {

  char c;
  string use = "USE";
  int len = strlen(use);

  gfc_char_t *p = gfc_current_locus.nextc;
  // Fixme : p == NULL
  while(!(char)*p == '\0')
    p++;
  char use_stmt[len + (p - gfc_current_locus.nextc) + 2];

  strcpy(use_stmt, use);
  p = gfc_current_locus.nextc;
  int pos = len;
  do {
    if(p == NULL) {
      c = '\0';
    } else {
      c = *p++;
    }

    use_stmt[pos++] = c;
  } while(c != '\0');
  if(ns2use == NULL) {
    ns2use = hash_table_make(hash_pointer, 0);
  }
  list use_stmts;
  if((use_stmts = hash_get(ns2use, (char *)gfc_current_ns))
      == HASH_UNDEFINED_VALUE) {
    use_stmts = CONS(string, strdup(use_stmt), NIL );
    hash_put(ns2use, (char *)gfc_current_ns, (char *)use_stmts);
  } else {
    CONS(string, strdup(use_stmt), use_stmts );
  }

}

list get_use_entities_list(struct gfc_namespace *ns) {
  list use_entities = NULL;
  if(ns2use) {
    int currentUse = 1;
    list use_stmts = NULL; // List of entities
    if((use_stmts = hash_get(ns2use, (char *)ns)) != HASH_UNDEFINED_VALUE) {
      string use = NULL;
      int current_len = 1;
      FOREACH(string, a_use, use_stmts) {
        int a_len = strlen(a_use);
        current_len += a_len;
        if(use == NULL) {
          use = (string)malloc(current_len * sizeof(char));
        } else {
          use = (string)realloc((void*)use, current_len * sizeof(char));
        }
        strcpy(&(use[current_len - a_len - 1]), a_use);
      }
      /* Create an entity */
      string entity_name;
      asprintf(&entity_name, "%s-use-%d", CurrentPackage, currentUse++);
      entity e = FindOrCreateEntity(F95_USE_LOCAL_NAME, use);
      entity_type(e) = make_type_unknown();
      entity_storage(e) = make_storage_rom();
      entity_initial(e) = make_value_unknown();
      use_entities = CONS(ENTITY,e,use_entities);
    }
  }
  return use_entities;
}


gfc_code* gfc2pips_get_last_loop(void) {
  if(gfc2pips_list_of_loops)
    return gfc2pips_list_of_loops->car.e;
  return NULL;
}
void gfc2pips_push_loop(gfc_code *c) {
  gfc2pips_list_of_loops = gen_cons(c, gfc2pips_list_of_loops);
}
void gfc2pips_pop_loop(void) {
  POP( gfc2pips_list_of_loops );
}

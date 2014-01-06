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
  * NewGen interface with C3 type Pvecteur for PIPS project 
  *
  * Remi Triolet
  *
  * Bugs:
  *
  *  - the NewGen interface makes it very cumbersome; function f()
  *    prevents a very simple fscanf() with format %d and %s and
  *    implies a copy in a temporary buffer (Francois Irigoin)
  *  - fixed for '\ )' in tokens (FC 2001/07/06)
  */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "linear.h"
#include "genC.h"
#include "ri.h"
#include "misc.h"

#define TCST_OLD_NAME "TERME_CONSTANT" /* compatibility to read old bases */
#define TCST_NAME     "."

/* print token */
static void print_token(FILE * fd, string s)
{
  for (; *s; s++)
  {
    /* backslashify '\ )' */
    if (*s=='\\' || *s==' ' || *s==')')
    {
      putc('\\', fd);
    }
    putc(*s, fd);
  }
}

/* stop at ' ' or ')'.
 * handles '\' as a protection.
 * returns a pointer to a static buffer.
 */
static string read_token(int (*f)())
{
  /* static internal buffer */
  static string buf = NULL;
  static int bufsize = 0;
  int index = 0, c;

  if (!buf)
  {
    bufsize = 64; /* should be ok for most codes. */
    buf = (char*) malloc(bufsize * sizeof(char));
    pips_assert("malloc ok", buf);
  }

  while ((c = f()) != -1 && c!=' ' && c!=')')
  {
    if (index+1>=bufsize) /* for current char and ending 0 */
    {
      bufsize *= 2;
      buf = (char*) realloc(buf, bufsize * sizeof(char));
      pips_assert("realloc ok", buf);
    }

    if (c == '\\')
    {
      c = f();
      pips_assert("there is a char after a backslash", c!=-1);
    }

    buf[index++] = (char) c;
  }
  buf[index++] = '\0';

  return buf;
}

/* output is "([val var ]* )"
 */
void vect_gen_write(FILE *fd, Pvecteur v)
{
    Pvecteur p;

    putc('(', fd);
    for (p = v; p != NULL; p = p->succ)
    {
      fprint_Value(fd, val_of(p));
      putc(' ', fd);
      print_token(fd, (p->var == (Variable) 0) ? TCST_NAME :
		  entity_name((entity) p->var));
      putc(' ', fd);
    }
    putc(')', fd);
}

Pvecteur vect_gen_read(FILE * fd __attribute__ ((unused)),
		       int (*f)())
{
  Pvecteur p = NULL;
  string svar, sval;
  Variable var;
  Value val;
  int openpar;

  openpar = f();
  pips_assert("vect starts with '('", openpar=='(');

  while ((sval = read_token(f)) && *sval)
  {
    sscan_Value(sval, &val);
    svar = read_token(f);

    if (same_string_p(svar, TCST_NAME) ||
	same_string_p(svar, TCST_OLD_NAME))
    {
      var = (Variable) 0;
    }
    else 
    {
      var = (Variable) gen_find_tabulated(svar, entity_domain);
      pips_assert("valid variable entity", !entity_undefined_p((entity)var));
    }
    vect_add_elem(&p, var, val);
  }

  p = vect_reversal(p);
  return p;
}

void vect_gen_free(Pvecteur v)
{
    vect_rm(v);
}

Pvecteur vect_gen_copy_tree(Pvecteur v)
{
    return vect_copy(v);
}

int vect_gen_allocated_memory(Pvecteur v)
{
    int result = 0;
    for (; v; v=v->succ)
	result += sizeof(Svecteur);
    return result;
}

int contrainte_gen_allocated_memory(Pcontrainte pc)
{
    int result = 0;
    for(; pc; pc=pc->succ)
	result += sizeof(Scontrainte) +
	    vect_gen_allocated_memory(pc->vecteur);
    return result;
}

/*   That is all
 */

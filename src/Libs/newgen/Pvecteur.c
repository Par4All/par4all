 /* NewGen interface with C3 type Pvecteur for PIPS project 
  *
  * Remi Triolet
  *
  * Bugs:
  *  - the NewGen interface makes it very cumbersome; function f()
  *    prevents a very simple fscanf() with format %d and %s and
  *    implies a copy in a temporary buffer (Francois Irigoin)
  */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "linear.h"
#include "genC.h"
#include "ri.h"
#include "misc.h"

#define TCST_NAME "TERME_CONSTANT"

static void print_token(FILE * fd, string s)
{
  for (; *s; s++)
  {
    if (*s=='\\' || *s==' ' || *s=='\n' || *s=='\0')
    {
      putc('\\', fd);
    }
    putc(*s, fd);
  }
}

/* as bad as strtok. NULL if empty string? */
static string read_token(string * ps)
{
  string head, r, w;
  if (!ps || !(*ps) || !(**ps)) return NULL;
  head = *ps, r=*ps, w=*ps;
  while (*r && *r!=' ')
  {
    if (*r=='\\')
    {
      r++;
      if (!*r) abort();
    }
    *w++ = *r++;
  }
  if (*r) *ps = r+1; 
  else *ps = NULL;
  *w = '\0';
  return head;
}

void vect_gen_write(FILE *fd, Pvecteur v)
{
    Pvecteur p;

    for (p = v; p != NULL; p = p->succ) {
	fprint_Value(fd, val_of(p));
	putc(' ', fd);
	print_token(fd, (p->var == (Variable) 0) ? TCST_NAME : 
		    entity_name((entity) p->var));
	putc(' ', fd);
    }
    putc('\n', fd);
}

Pvecteur vect_gen_read(fd, f)
FILE *fd; /* ignored */
int (*f)();
{
    static char * buffer = NULL;
    static int buffersize = 1024;
    Value val;
    Variable var;
    char *varname, *sval;
    char *pbuffer;
    int ibuffer = 0;
    Pvecteur p = NULL;
    int c, previous=0;

    if (!buffer) 
    {
      buffer = (char*) malloc(buffersize*sizeof(char));
      pips_assert("malloc ok", buffer);
    }

    /* read buffer up to new line */
    while ((c = f()) != -1 && (c!='\n' && previous!='\\'))
    {
      if (ibuffer+1>=buffersize)
      {
	buffersize*=2;
	buffer = (char*) realloc(buffer, buffersize*sizeof(char));
	pips_assert("realloc ok", buffer);
      }
      buffer[ibuffer++] = c;
      previous = (previous=='\\')? 0 : c;
    }

    buffer[ibuffer++] = '\0';

    pbuffer = buffer;
    sval = read_token(&pbuffer); 
    while (sval != NULL) 
    {
      sscan_Value(sval, &val);
      varname = read_token(&pbuffer);
      if (strcmp(varname, TCST_NAME) == 0) {
	var = (Variable) 0;
      }
      else {
	var = (Variable) gen_find_tabulated(varname, entity_domain);
	if (var == (Variable) entity_undefined) {
	  fprintf(stderr, "[vect_gen_read] bad entity name: %s\n",
		  varname);
	  abort();
	}
      }
      
      vect_add_elem(&p, var, val);
      
      /* pbuffer = strtok(NULL, " "); */
      sval = read_token(&pbuffer); 
    }

    p = vect_reversal(p);
    return(p);
}

void vect_gen_free(Pvecteur v)
{
    vect_rm(v);
}

Pvecteur vect_gen_copy_tree(Pvecteur v)
{
    return vect_dup(v);
}

int 
vect_gen_allocated_memory(
    Pvecteur v)
{
    int result = 0;
    for (; v; v=v->succ)
	result += sizeof(Svecteur);
    return result;
}

int
contrainte_gen_allocated_memory(
    Pcontrainte pc)
{
    int result = 0;
    for(; pc; pc=pc->succ)
	result += sizeof(Scontrainte) + 
	    vect_gen_allocated_memory(pc->vecteur);
    return result;
}

/*   That is all
 */

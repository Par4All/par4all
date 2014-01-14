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

#include <stdio.h>


//we need to copy the content of the locus
void gfc2pips_push_comment(locus l, unsigned long num, char s) {
  printf("gfc2pips_push_comment \n");
  if(gfc2pips_comments_stack) {
    if(gfc2pips_check_already_done(l)) {
      return;
    }
    gfc2pips_comments_stack->next = malloc(sizeof(struct _gfc2pips_comments_));
    gfc2pips_comments_stack->next->prev = gfc2pips_comments_stack;
    gfc2pips_comments_stack->next->next = NULL;

    gfc2pips_comments_stack = gfc2pips_comments_stack->next;
  } else {
    gfc2pips_comments_stack = malloc(sizeof(struct _gfc2pips_comments_));
    gfc2pips_comments_stack->prev = NULL;
    gfc2pips_comments_stack->next = NULL;
    gfc2pips_comments_stack_ = gfc2pips_comments_stack;
  }
  //fprintf(stderr,"push comments %d\n",l.lb->location);

  gfc2pips_comments_stack->l = l;
  gfc2pips_comments_stack->num = num;
  gfc2pips_comments_stack->gfc = NULL;
  gfc2pips_comments_stack->done = false;

  gfc2pips_comments_stack->s = gfc2pips_gfc_char_t2string2(l.nextc);
  gfc2pips_comments_stack->s[strlen(gfc2pips_comments_stack->s) - 2] = '\0';
  strrcpy(gfc2pips_comments_stack->s + 1, gfc2pips_comments_stack->s);
  *gfc2pips_comments_stack->s = s;
  printf("gfc2pips_push_comment : '%s'\n", gfc2pips_comments_stack->s);
}

bool gfc2pips_check_already_done(locus l) {
  gfc2pips_comments retour = gfc2pips_comments_stack;
  while(retour) {
    if(retour->l.nextc == l.nextc)
      return true;
    retour = retour->prev;
  }
  return false;
}


unsigned long gfc2pips_get_num_of_gfc_code(gfc_code *c) {
  unsigned long retour = 0;
  gfc2pips_comments curr = gfc2pips_comments_stack_;
  while(curr) {
    if(curr->gfc == c) {
      return retour + 1;
    }
    curr = curr->next;
    retour++;
  }
  if(retour)
    return retour + 1;
  return retour;// 0
}
string gfc2pips_get_comment_of_code(gfc_code *c) {
  gfc2pips_comments retour = gfc2pips_comments_stack_;
  char *a, *b;
  while(retour) {
    if(retour->gfc == c) {
      a = retour->s;
      retour = retour->next;
      while(retour && retour->gfc == c) {
        if(a && retour->s) {
          b = (char*)malloc(sizeof(char) * (strlen(a) + strlen(retour->s) + 2));
          strcpy(b, a);
          strcpy(b + strlen(b), "\n");
          strcpy(b + strlen(b), retour->s);
          free(a);
          a = b;
        } else if(retour->s) {
          a = retour->s;
        }
        retour = retour->next;
      }
      if(a) {
        b = (char*)malloc(sizeof(char) * (strlen(a) + 2));
        strcpy(b, a);
        strcpy(b + strlen(b), "\n");
        free(a);
        return b;
      } else {
        return empty_comments;
      }
    }
    retour = retour->next;
  }
  return empty_comments;
}

gfc2pips_comments gfc2pips_pop_comment(void) {
  if(gfc2pips_comments_stack) {
    gfc2pips_comments retour = gfc2pips_comments_stack;
    gfc2pips_comments_stack = gfc2pips_comments_stack->prev;
    if(gfc2pips_comments_stack) {
      gfc2pips_comments_stack->next = NULL;
    } else {
      gfc2pips_comments_stack_ = NULL;
    }
    return retour;
  } else {
    return NULL;
  }
}

//changer en juste un numéro, sans que ce soit "done"
//puis faire une étape similaire qui assigne un statement à la première plage non "done" et la met à "done"
void gfc2pips_set_last_comments_done(unsigned long nb) {
  //printf("gfc2pips_set_last_comments_done\n");
  gfc2pips_comments retour = gfc2pips_comments_stack;
  while(retour) {
    if(retour->done)
      return;
    retour->num = nb;
    retour->done = true;
    retour = retour->prev;
  }
}
void gfc2pips_assign_num_to_last_comments(unsigned long nb) {
  gfc2pips_comments retour = gfc2pips_comments_stack;
  while(retour) {
    if(retour->done || retour->num)
      return;
    retour->num = nb;
    retour = retour->prev;
  }
}
void gfc2pips_assign_gfc_code_to_last_comments(gfc_code *c) {
  gfc2pips_comments retour = gfc2pips_comments_stack_;
  if(c) {
    while(retour && retour->done) {
      retour = retour->next;
    }
    if(retour) {
      unsigned long num_plage = retour->num;
      while(retour && retour->num == num_plage) {
        retour->gfc = c;
        retour->done = true;
        retour = retour->next;
      }
    }
  }
}

void gfc2pips_replace_comments_num(unsigned long old, unsigned long new) {
  gfc2pips_comments retour = gfc2pips_comments_stack;
  bool if_changed = false;
  //fprintf(stderr,"gfc2pips_replace_comments_num: replace %d by %d\n", old, new );
  while(retour) {
    if(retour->num == old) {
      if_changed = true;
      retour->num = new;
    }
    retour = retour->prev;
  }
  //if(if_changed) gfc2pips_nb_of_statements--;
}

void gfc2pips_assign_gfc_code_to_num_comments(gfc_code *c, unsigned long num) {
  gfc2pips_comments retour = gfc2pips_comments_stack_;
  while(retour) {
    if(retour->num == num)
      retour->gfc = c;
    retour = retour->next;
  }
}
bool gfc2pips_comment_num_exists(unsigned long num) {
  gfc2pips_comments retour = gfc2pips_comments_stack;
  //fprintf(stderr,"gfc2pips_comment_num_exists: %d\n", num );
  while(retour) {
    if(retour->num == num)
      return true;
    retour = retour->prev;
  }
  return false;
}

void gfc2pips_pop_not_done_comments(void) {
  while(gfc2pips_comments_stack && gfc2pips_comments_stack->done == false) {
    gfc2pips_pop_comment();
  }
}

/**
 *  \brief We assign a gfc_code depending to a list of comments if any
 *  depending on the number of the statement
 */
void gfc2pips_shift_comments(void) {
  /*
   *
   */
  gfc2pips_comments retour = gfc2pips_comments_stack;
  list l = gen_nreverse(gfc2pips_list_of_declared_code);
  while(retour) {

    list curr = gen_nthcdr(retour->num, l);
    if(curr) {
      retour->gfc = (gfc_code*)curr->car.e;
    }
    retour = retour->prev;
  }
  return;
}

void gfc2pips_push_last_code(gfc_code *c) {
  if(gfc2pips_list_of_declared_code == NULL)
    gfc2pips_list_of_declared_code = gen_cons(NULL, NULL);
  //gfc2pips_list_of_declared_code =
  gfc2pips_list_of_declared_code = gen_cons(c, gfc2pips_list_of_declared_code);
}



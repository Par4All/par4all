/*
  $Id$

  Copyright 1989-2014 MINES ParisTech

  This file is part of Linear/C3 Library.

  Linear/C3 Library is free software: you can redistribute it and/or modify it
  under the terms of the GNU Lesser General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  Linear/C3 Library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public License
  along with Linear/C3 Library.  If not, see <http://www.gnu.org/licenses/>.

*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <getopt.h>

#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

extern char * sc_internal_symbol_table(char *);

int main(int argc, char * const argv[])
{
  // option management
  bool reverse = false;
  int debug = 0;
  int opt;
  while ((opt = getopt(argc, argv, "rD")) != -1) {
    switch (opt) {
    case 'r': reverse = true; break;
    case 'D': debug++; break;
    default: exit(1);
    }
  }
  // get system
  Psysteme s;
  bool ok = sc_fscan(stdin, &s);
  assert(ok);
  if (debug >= 2) sc_fprint(stderr, s, *variable_default_name);
  // project & simplify
  if (reverse) {
    Pbase bnext;
    for (Pbase b = sc_base(s); b && s; b = bnext) {
      // Oops: the system may be nullifed by sc_projection...
      // so the next iteration must not rely on it
      bnext = vecteur_succ(b);
      bool keep = false;
      for (int i = optind; i < argc; i++) {
        if (strcmp(argv[i], vecteur_var(b)) == 0) {
          keep = true;
          break;
        }
      }
      if (!keep)
        s = sc_projection(s, vecteur_var(b));
    }
  }
  else {
    for(int i = optind; i < argc; i++)
      sc_projection(s, sc_internal_symbol_table(argv[i]));
  }
  if (debug) sc_fprint(stderr, s, *variable_default_name);
  sc_nredund(&s);
  // print out result
  sc_fprint(stdout, s, *variable_default_name);
  return 0;
}

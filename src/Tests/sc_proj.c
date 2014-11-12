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
  bool reverse = false, bound = false;
  int debug = 0, opt;
  while ((opt = getopt(argc, argv, "bhrD")) != -1) {
    switch (opt) {
    case 'r': reverse = true; break;
    case 'b': bound = true; break;
    case 'D': debug++; break;
    case 'h':
      fprintf(stdout,
              "usage: %s [-r or -b] variables... < system\n"
              "\tdefault: project listed variables\n"
              "\t-r: project all but the listed variables\n"
              "\t-b: show bounds on listed variables\n",
              argv[0]);
      exit(0);
    default: exit(1);
    }
  }
  // only one of -r and -b
  assert(!(reverse && bound));

  // get system from stdin
  Psysteme s;
  bool ok = sc_fscan(stdin, &s);
  assert(ok);
  if (debug >= 2) sc_fprint(stderr, s, *variable_default_name);

  // get command arguments as a base
  Pbase arg_base = BASE_NULLE;
  for (int i = optind; i < argc; i++)
    arg_base = base_add_variable(arg_base, sc_internal_symbol_table(argv[i]));

  Pbase proj_base;
  if (reverse || bound) {
    proj_base = vect_copy(sc_base(s));
    for (Pbase b = arg_base; b != BASE_NULLE; b = vecteur_succ(b))
      proj_base = base_remove_variable(proj_base, vecteur_var(b));
  }
  else {
    proj_base = vect_copy(arg_base);
  }

  // project all "proj_base" variables
  for (Pbase b = proj_base; b != BASE_NULLE; b = vecteur_succ(b))
      s = sc_projection(s, vecteur_var(b));

  // cleanup
  sc_nredund(&s);

  if (debug) sc_fprint(stderr, s, *variable_default_name);

  // print out result
  if (bound) {
    // we must bound each remaining variables
    for (Pbase b = arg_base; b != BASE_NULLE; b = vecteur_succ(b)) {
      Variable keep = vecteur_var(b);
      Psysteme sb = sc_copy(s);
      // YES, we use *s* base
      for (Pbase c = sc_base(s); c != BASE_NULLE; c = vecteur_succ(c)) {
        if (vecteur_var(c) != keep)
          sb = sc_projection(sb, vecteur_var(c));
      }
      sc_nredund(&sb);
      sc_fprint(stdout, sb, *variable_default_name);
      sc_rm(sb);
    }
  }
  else
    sc_fprint(stdout, s, *variable_default_name);

  base_rm(arg_base);
  base_rm(proj_base);
  sc_rm(s);
  return 0;
}

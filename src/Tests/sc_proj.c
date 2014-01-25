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
#include <assert.h>

#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

extern char * sc_internal_symbol_table(char *);

int main(int argc, char * argv[])
{
  Psysteme s;
  bool ok = sc_fscan(stdin, &s);
  assert(ok);
  for(int i = 1; i < argc; i++)
    sc_projection(s, sc_internal_symbol_table(argv[i]));
  sc_fprint(stdout, s, *variable_default_name);
  return 0;
}

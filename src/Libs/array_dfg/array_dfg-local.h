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
/* local definitions of this library */
#define RETURN( _debug, _function, _ret_obj ) \
   { debug((_debug), (_function), "returning \n"); return( (_ret_obj) ); }
#define ADFG_MODULE_NAME        "ADFG"
#define EXPRESSION_PVECTEUR(e) \
    (normalized_linear(NORMALIZE_EXPRESSION( e )))
#define ENTRY_ORDER     300000  /* Hard to have a non illegal number for hash_put !!! */
#define EXIT_ORDER      100000
#define TAKE_LAST	TRUE
#define TAKE_FIRST	FALSE

/* Structure for return of a possible source */
typedef struct Sposs_source {
        quast   *qua;
        Ppath   pat;
} Sposs_source, *Pposs_source;


/* Structure to list wich node read or write an effect */
typedef struct Sentity_vertices {
        entity ent;
        list lis;    /* list of vertices */
} Sentity_vertices, *Pentity_vertices;


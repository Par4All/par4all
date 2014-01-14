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

/*
* includes for newgen
* a link to the newgen binary file is missing
*/

#ifndef GFC_2_PIPS
#define GFC_2_PIPS


#include "config.h"
#include "system.h"
#include "gfortran.h"

/*
//hack : conflict for list which is defined as a type in PIPS and as a variable in gfc
#undef list
#define list newgen_list

//hack : bool/boolean defined in both gfc and PIPS and in different ways
*/


//#define string  char *


//hack : match
#define match pips_match
#define hash_table pips_hash_table
#define loop pips_loop

//hack : After gfortran.h free become a macro for another function
#undef free







#undef loop
#undef hash_table
#undef match
#undef list

extern int gfc2pips_nb_of_statements;


typedef struct _gfc2pips_comments_{
	unsigned char done; // Bool
	locus l;
	gfc_code *gfc;
	char *s;
	unsigned long num;
	struct _gfc2pips_comments_ *prev;
	struct _gfc2pips_comments_ *next;
} *gfc2pips_comments;


extern gfc2pips_comments gfc2pips_comments_stack;
extern gfc2pips_comments gfc2pips_comments_stack_;
extern bool get_bool_property(const string );


void gfc2pips_set_last_comments_done(unsigned long nb);
void gfc2pips_pop_not_done_comments(void);
void gfc2pips_replace_comments_num(unsigned long old, unsigned long new);


#endif /* GFC_2_PIPS */


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

%option nounput
%option noinput

%{
//%option debug
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
#include <stdlib.h>
#include <string.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"

#include "resources.h"

#include "ri-util.h"
#include "sac.h"
#include "patterns.h"

%}
%%
"REFERENCE"               return REFERENCE_TOK;
"QI_REF"                return QI_REF_TOK;
"HI_REF"                return HI_REF_TOK;
"SI_REF"                return SI_REF_TOK;
"DI_REF"                return DI_REF_TOK;
"SF_REF"                return SF_REF_TOK;
"DF_REF"                return DF_REF_TOK;
"SC_REF"                return SC_REF_TOK;
"DC_REF"                return DC_REF_TOK;
"LOG_REF"               return LOG_REF_TOK;
"CONSTANT"              return CONSTANT_TOK;
"="			return ASSIGN_OPERATOR_TOK;
"+"			return PLUS_OPERATOR_TOK;
"-"			return MINUS_OPERATOR_TOK;
"UNARY_MINUS"		return UNARY_MINUS_OPERATOR_TOK;
"*"      		return MULTIPLY_OPERATOR_TOK;
"__PIPS_SAC_MULADD"	return MULADD_OPERATOR_TOK;
"/"			return DIVIDE_OPERATOR_TOK;
"INVERSE"		return INVERSE_OPERATOR_TOK;
"^"			return POWER_OPERATOR_TOK;
"%"			return MODULO_OPERATOR_TOK;
"MIN"			return MIN_OPERATOR_TOK;
"COS"			return COS_OPERATOR_TOK;
"SIN"			return SIN_OPERATOR_TOK;
"MIN0"			return MIN0_OPERATOR_TOK;
"AMIN1"			return AMIN1_OPERATOR_TOK;
"DMIN1"			return DMIN1_OPERATOR_TOK;
"MAX"			return MAX_OPERATOR_TOK;
"MAX0"			return MAX0_OPERATOR_TOK;
"AMAX1"			return AMAX1_OPERATOR_TOK;
"DMAX1"			return DMAX1_OPERATOR_TOK;
"ABS"			return ABS_OPERATOR_TOK;
"IABS"			return IABS_OPERATOR_TOK;
"DABS"			return DABS_OPERATOR_TOK;
"CABS"			return CABS_OPERATOR_TOK;
"&"			return AND_OPERATOR_TOK;
"|"			return OR_OPERATOR_TOK;
"!"			return NOT_OPERATOR_TOK;
"!="     		return NON_EQUAL_OPERATOR_TOK;
"EQUIV"			return EQUIV_OPERATOR_TOK;
"NON_EQUIV"		return NON_EQUIV_OPERATOR_TOK;
"TRUE"			return TRUE_OPERATOR_TOK;
"FALSE"			return FALSE_OPERATOR_TOK;
">="            return GREATER_OR_EQUAL_OPERATOR_TOK;
">"     		return GREATER_THAN_OPERATOR_TOK;
"<="		    return LESS_OR_EQUAL_OPERATOR_TOK;
"<"      		return LESS_THAN_OPERATOR_TOK;
"=="			return EQUAL_OPERATOR_TOK;
"?"				return PHI_TOK;
[0-9.-]+                  { yylval.iVal = atoi(yytext); return INTEGER_TOK; }
[A-Za-z][0-9A-Za-z_]*   { yylval.strVal = strdup(yytext); return IDENTIFIER_TOK; }
[ \t]                   ;
[\n]					{ sac_lineno ++; }
"#".*                   ;
.                       return *yytext;
%%

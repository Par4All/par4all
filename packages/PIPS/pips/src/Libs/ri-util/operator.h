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
/*
 * FI: these values were taken from syntax/tokyacc.h but I do not
 * think they matter. Wouldn't an enum be better?
 *
 * MOD was an initial exception. So are MINIMUM and MAXIMUM
 */

#define AND     55
#define EQ     56
#define EQV     57
#define GE     58
#define GT     59
#define LE     60
#define LT     61
#define NE     62
#define NEQV     63
#define NOT     64
#define OR     65
#define MINUS     73
#define PLUS     74
#define SLASH     75
#define STAR     76
#define POWER     77
#define MOD       78  /* not evaluated, but later added in IsBinaryOperator*/
#define CONCAT     84
#define MINIMUM 85
#define MAXIMUM 86
#define CAST_OP 87
#define BITWISE_AND 88
#define BITWISE_OR 89
#define BITWISE_XOR 90
#define RIGHT_SHIFT 91
#define LEFT_SHIFT 92

#define ASSIGN 100
#define POST_INCREMENT 101
#define POST_DECREMENT 102
#define PRE_INCREMENT 103
#define PRE_DECREMENT 104
#define MULTIPLY_UPDATE 105
#define DIVIDE_UPDATE 106
#define PLUS_UPDATE 107
#define MINUS_UPDATE 108
#define LEFT_SHIFT_UPDATE 109
#define RIGHT_SHIFT_UPDATE 110
#define BITWISE_OR_UPDATE 111

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
/* -- control.h */

#ifndef CONTROL_INCLUDED
#define CONTROL_INCLUDED

/* Define the environment variables used to select which controlizer
   version to use independently of which one was activated at the pipsmake
   level: */
/* The name of the one to force the use of the new controlizer: */
#define USE_NEW_CONTROLIZER_ENV_VAR_NAME "PIPS_USE_NEW_CONTROLIZER"
/* The name of the one to force the use of the old controlizer: */
#define USE_OLD_CONTROLIZER_ENV_VAR_NAME "PIPS_USE_OLD_CONTROLIZER"
#endif

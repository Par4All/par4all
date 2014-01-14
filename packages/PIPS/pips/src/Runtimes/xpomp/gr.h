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
enum gr_flag {
GR_START,
GR_BUF,
GR_MOUSE,
GR_ISMOUSE,
GR_CMAP,
GR_PUT_FRAME,
GR_SET_COLOR,
GR_SCROLL,
GR_CLOSE,
GR_BUF_CLIP
};
struct sh_header{
unsigned int lock:1;
unsigned int activite:1;
enum gr_flag flag;
int bufsize;
int id;
int p1;
int p2;
int p3;
int p4;
int p5;
int p6;
int p7;
int p8;
int p9;
int p10;
char *buf1;
char *buf2;
};

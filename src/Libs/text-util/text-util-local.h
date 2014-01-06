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

#define FORESYS_CONTINUATION_PREFIX "C$&" "    "

#define MAKE_SWORD(s) strdup(s)
#define MAKE_IWORD(i) i2a(i)
#define MAKE_FWORD(f) f2a(f)

#define CHAIN_SWORD(l,s) gen_nconc(l, CONS(STRING, MAKE_SWORD(s), NIL))
#define CHAIN_IWORD(l,i) gen_nconc(l, CONS(STRING, MAKE_IWORD(i), NIL))
#define CHAIN_FWORD(l,f) gen_nconc(l, CONS(STRING, MAKE_FWORD(f), NIL))

#define MAKE_ONE_WORD_SENTENCE(m, s)					\
  make_sentence(is_sentence_unformatted,				\
		make_unformatted((char *) NULL, 0, m, CHAIN_SWORD(NIL, s)))

#define ADD_SENTENCE_TO_TEXT(t,p)					\
  do {									\
    text _t_ = (t);							\
    text_sentences(_t_) =						\
      gen_nconc(text_sentences(_t_), CONS(SENTENCE, (p), NIL));		\
  } while(0)

#define MERGE_TEXTS(r,t)					\
  do {								\
    text _r_ = (r); text _t_ = (t);				\
    text_sentences(_r_) =					\
      gen_nconc(text_sentences(_r_), text_sentences(_t_));	\
    text_sentences(_t_) = NIL;					\
    free_text(_t_);						\
  } while(0)

/* maximum length of a line when prettyprinting...
 * from 0 to 69, i.e. 70 chars, plus "\n\0"
 */
#define MAX_LINE_LENGTH 72

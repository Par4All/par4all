/*
 * $Id$
 */

#define PIPS_COMMENT_SENTINEL 		"C"
#define PIPS_COMMENT_PREFIX   		PIPS_COMMENT_SENTINEL
#define PIPS_COMMENT_CONTINUATION 	PIPS_COMMENT_SENTINEL "    "

#define FORESYS_CONTINUATION_PREFIX "C$&" "    "

#define MAKE_SWORD(s) strdup(s)
#define MAKE_IWORD(i) i2a(i)
#define MAKE_FWORD(f) f2a(f)

#define CHAIN_SWORD(l,s) gen_nconc(l,CONS(STRING, MAKE_SWORD(s), NIL))
#define CHAIN_IWORD(l,i) gen_nconc(l,CONS(STRING, MAKE_IWORD(i), NIL))
#define CHAIN_FWORD(l,f) gen_nconc(l,CONS(STRING, MAKE_FWORD(f), NIL))

#define MAKE_ONE_WORD_SENTENCE(m, s) \
    make_sentence(is_sentence_unformatted, \
		  make_unformatted((char *) NULL, 0, m, CHAIN_SWORD(NIL, s)))

#define ADD_SENTENCE_TO_TEXT(t,p) { text _t_ = (t); \
 text_sentences(_t_)=gen_nconc(text_sentences(_t_), CONS(SENTENCE, (p), NIL));}

#define MERGE_TEXTS(r,t) { \
       text _r_ = (r); text _t_ = (t); \
       text_sentences(_r_) = \
	   gen_nconc(text_sentences(_r_), text_sentences(_t_)); \
       text_sentences(_t_) = NIL; \
       free_text(_t_); \
       }

/* maximum length of a line when prettyprinting...
 * from 0 to 69, i.e. 70 chars, plus "\n\0" 
 */
#define MAX_LINE_LENGTH 72

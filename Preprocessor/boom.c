 void f(char* s) {
}

void boom() {
  /* Stress a bug found by Johan Gall while analyzing PERL from the
   SPEC2006. */
  f("\\");
  f("abc");
  f(L"\\");
  f(L"abc");
  f("\\");
  f(L"\\");
  f("abc");
  f(L"abc");

  /* Note this was a special cas in the parser, but according to the C norm
     (WG14/N1256 Committee Draft - Septermber 7, 2007 ISO/IEC 9899:TC3,
     Annex 1.1.6) I think it should not. It is indead the concatenation of
     2 empty strings. RK. */
  f("""");
  f("" "");
  /* But I wonder what is the meaning of this: the concatenation of a
     wide-char string with a normal-char string. I guess it should not be
     legal...  */
  f(L"""");
  f(L"" "");
  /* Well, indeed, it correspond to the comment in the parser ``the
     "world" in L"Hello, " "world" should be treated as wide even though
     there's no L immediately preceding it'' */

  /* But this is not legal I guess (especially for this on word-aligned
     memory architecture): */
  /* f("abc" L"d"); */
}

/* Test litteral character constants: */ 
int main () {
  boom ();
  return 0;
}

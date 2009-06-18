#include <stddef.h>

/* Test litteral character constants: */
int main () {
  char *s;
  wchar_t *ws;
  unsigned char c = '\023';
  wchar_t wc  = L'\076';
  c = '\0';
  c = '\01';
  c = '\xc';
  c = '\xef';
  wc = '\xde';
  wc = L'\xdef';
  wc = L'\xabcdef01';
  wc = L'\U0001babe';
  wc = L'\ubabe';

  s = "\023\076\0\01\xc\xef\xde";

  ws = L"\023\076\0\01\xc\xef\xde\xdef\xabcdef01\U0001babe\ubabe";

  return 0;
}

/* The extern declaration is invalidated by the direct declaration */

extern int a;
int a;

int b;
extern int b;
int b;

// The prettyprinter should memorize declared and initialized
// variables so as not to redeclare them.
//int c = 3;
int c;
extern int c;

int main() {
  a = 9;
  b = 8;
  return a+b+c;
}

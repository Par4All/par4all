/* This is fully ANSI compatible because the scopes are different for
   each a and for each f */

/* Example by Ronan */
void decl09(int a, void (*f)(char a, void (*f)(char a, void (*f)(char a ))));

/* Early output by PIPS: */
void decl09b(int ,void (*)(char ,void (*)(char ,void (*)(char ))));

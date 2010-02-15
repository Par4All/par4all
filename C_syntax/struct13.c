/* This code is incorrect because struct s is redeclared */
struct s {
   int l;
};
extern void struct13(union {struct s {int l;} d1; struct s {int l;} d2; int i;} u);

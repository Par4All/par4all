/* Example by Ronan keryell */
typedef int f(int cookie);

typedef int (*g)(int);

f decl19;

int decl19(int cookie) {
  g x;
  x = decl19;
  //x = &main;
  /* This is not possible :-)
     x = &&&&&main; */
  if (cookie < 10)
    return x(cookie + 1);
  else if (cookie < 20 )
    return (*x)(cookie + 1);
  else if (cookie < 30 )
  /* Fun for my student :-) */
  (*****************x)(cookie + 1);
}

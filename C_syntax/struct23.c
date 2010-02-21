/* Bug in preprocessor because of "s" in "struct s struct23()" before
   "struct23" (Ticket 298) */

struct s {
  int in;
};

struct s struct23()
{
  struct s s;
  s.in = 0;
  return s;
}

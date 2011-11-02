// Really weird code submitted by Serge Guelton, not likely to survive
// a gcc -c -Wall -Werror. Besides, the options used for PIPS means it
// should have been places in Semantics-New. return added in
// hs_set_r() to display precondition resulting from call to
// vhs_set_r().
//
// Issues potentially leading to an unusual control path in PIPS:
//
// 1. Typedef used only to declare an enum
//
// 2. Useless argument o for vhs_set_r()
//
// 3. Unitialized variable o in hs_set_r()
//
// See bug description below

typedef enum a {
  HS_PARSE_ONLY,
} ;

int vhs_set_r(enum a o) {
  return 1;
}

/* replacing enum a by an int works fine */
void hs_set_r() {
  enum a o;
  int res = vhs_set_r(o);
  return;
}

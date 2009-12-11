/* Bug encountered in SPEC2000/mesa.c: anonymous struct is not
   referenced by typedef EdgeT */

static void flat_color_z_triangle()
{
  {
    typedef struct {
      int v0;
    } EdgeT;

    EdgeT eMaj;

    eMaj.v0 = 1;
  }
}

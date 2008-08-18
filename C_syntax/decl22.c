/* Double declaration of a global constant in a compilation unit is OK, even with -ansi and -pedantic */

/* PIPS prettyprinter cannot debuild the source code properly */

const char * MAK_not_byte;

const char * MAK_not_byte = "L'image n'est pas du type BYTE";

/* A second initialization is not OK, even for gcc with no optionss beyong -c */
const char * MAK_not_byte = "L'image n'est pas du type BYTE";

int v;

int v;

/*
void decl22()
{
  int v;
  int v;
}
*/

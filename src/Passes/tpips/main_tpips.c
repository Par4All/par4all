/* 
 * $Id$
 *
 * This file contains the main for tpips.
 * Please, do not change anything! do any change to tpips_main().
 *
 * FC.
 */

extern char * pips_thanks(char *);
extern int tpips_main(int, char**);

int main(int argc, char ** argv)
{
    pips_thanks("tpips", argv[0]);
    return tpips_main(argc, argv);
}

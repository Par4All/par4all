/* tpips core dumps while handling this declaration in
 * ngspice_bug/main.tpips, the problem is not replicated here.
 */

    struct {
      //GRIDTYPE gridtype;        /* defined in FTEconstant.h */
      int circular;         /* TRUE if circular plot area */
      union {
	struct {
	  char units[16];     /* unit labels */
	  int	spacing, numspace;
	  double	distance, lowlimit, highlimit;
	  int	mult;
	  int	onedec;     /* a boolean */
	  int	hacked;     /* true if hi - lo already hacked up */
	  double	tenpowmag;
	  double	tenpowmagx;
	  int	digits;
	} lin;
	struct {
	  char units[16];     /* unit labels */
	  int hmt, lmt, decsp, subs, pp;
	} log;
	struct {
	  char units[16];     /* unit labels */
	  int radius, center;
	  double mrad;
	  int lmt;
	  int hmt, mag; /* added, p.w.h. */
	} circular;     /* bogus, rework when write polar grids, etc */
      } xaxis, yaxis;
      int xdatatype, ydatatype;
      int xsized, ysized;
      double xdelta, ydelta; /* if non-zero, user-specified deltas */
      char *xlabel, *ylabel;
    } grid;

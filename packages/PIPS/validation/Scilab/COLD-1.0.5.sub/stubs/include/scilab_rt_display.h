/* (c) HPC Project 2010 */

extern void scilab_rt_display_s0i0_(char* name, int in0);

extern void scilab_rt_display_s0d0_(char* name, double in0);

extern void scilab_rt_display_s0z0_(char* name, double complex in0); 

extern void scilab_rt_display_s0s0_(char* name, char* in0); 

extern void scilab_rt_display_s0i2_(char* name, int si00, int si01, int in0[si00][si01]);

extern void scilab_rt_display_s0d2_(char* name, int si00, int si01, double in0[si00][si01]);

extern void scilab_rt_display_s0s2_(char* name, int si00, int si01, char* in0[si00][si01]);

extern void scilab_rt_display_s0z2_(char* name, int si00, int si01, double complex in0[si00][si01]);

extern void scilab_rt_display_s0i3_(char* name, int si00, int si01, int si02, int in0[si00][si01][si02]);

extern void scilab_rt_display_s0d3_(char* name, int si00, int si01, int si02, double in0[si00][si01][si02]);

extern void scilab_rt_display_s0z3_(char* name, int si00, int si01, int si02, double complex in0[si00][si01][si02]);

extern void scilab_rt_display_s0i4_(char* name, int si00, int si01, int si02, int si03, int in0[si00][si01][si02][si03]);

extern void scilab_rt_display_s0d4_(char* name, int si00, int si01, int si02, int si03, double in0[si00][si01][si02][si03]);

extern void scilab_rt_display_s0z4_(char* name, int si00, int si01, int si02, int si03, double complex in0[si00][si01][si02][si03]);


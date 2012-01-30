/*---------------------------------------------------- -*- C -*-
*
*   (c) HPC Project - 2010
*
*/

extern void scilab_rt_histplot__();

extern void scilab_rt_histplot_i0i2_(int scalarin0, 
    int in00, int in01, int matrixin0[in00][in01]);

extern void scilab_rt_histplot_i2i2_(int in00, int in01, int matrixin0[in00][in01], 
    int in10, int in11, int matrixin1[in10][in11]);

extern void scilab_rt_histplot_i0d2_(int scalarin0, 
    int in00, int in01, double matrixin0[in00][in01]);

extern void scilab_rt_histplot_i2d2_(int in00, int in01, int matrixin0[in00][in01], 
    int in10, int in11, double matrixin1[in10][in11]);

extern void scilab_rt_histplot_d0i2_(double scalarin0, 
    int in00, int in01, int matrixin0[in00][in01]);

extern void scilab_rt_histplot_d2i2_(int in00, int in01, double matrixin0[in00][in01], 
    int in10, int in11, int matrixin1[in10][in11]);

extern void scilab_rt_histplot_d0d2_(double scalarin0, 
    int in00, int in01, double matrixin0[in00][in01]);

extern void scilab_rt_histplot_d2d2_(int in00, int in01, double matrixin0[in00][in01], 
    int in10, int in11, double matrixin1[in10][in11]);


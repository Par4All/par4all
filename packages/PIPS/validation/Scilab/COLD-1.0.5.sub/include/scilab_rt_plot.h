/*---------------------------------------------------- -*- C -*-
*
*   (c) HPC Project - 2010
*
*/

extern void scilab_rt_plot__();

extern void scilab_rt_plot_i2_(int in00, int in01, int matrixin0[in00][in01]);

extern void scilab_rt_plot_d2_(int in00, int in01, double matrixin0[in00][in01]);

extern void scilab_rt_plot_i2i2_(int in00, int in01, int matrixin0[in00][in01], 
    int in10, int in11, int matrixin1[in10][in11]);

extern void scilab_rt_plot_i2d2_(int in00, int in01, int matrixin0[in00][in01], 
    int in10, int in11, double matrixin1[in10][in11]);

extern void scilab_rt_plot_d2i2_(int in00, int in01, double matrixin0[in00][in01], 
    int in10, int in11, int matrixin1[in10][in11]);

extern void scilab_rt_plot_d2d2_(int in00, int in01, double matrixin0[in00][in01], 
    int in10, int in11, double matrixin1[in10][in11]);


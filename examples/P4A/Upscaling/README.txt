This upscaling program was developed to test an architecture prototype
based on CPU and GPU and designed for multimedia transcoding and
processing inside core-network gateways. This work was achieved in the
TransMedi@ project framework from the French Images and Networks
research cluster. The upscaling function is the cubic one (6 points)
with a factor 2 (doubling the size of the image) taken from the H264
standard.  In takes a video in yuv format as input and displays the
upscaled video in yuv format as output.

Two versions : 

1) The first version of upscale_luminance in upscale.c
calculates y_out(i,j), y_out(i+1,j) and y_out(i,j+1) from y_in
(upscale_lunimance_centre) and y_out(i+1,j+1) from y_out
(upscale_lunimance_xplus1yplus1).

In this first version : call upscale.c in the Makefile 

2) The second version calculates all y_out(i,j), y_out(i+1,j),
y_out(i,j+1) and y_out(i+1,j+1) from y_in (upscale_luminance). This
version is more computation intensive but with more parallelism.

In this second version : call upscaleBis.c in the Makefile 

3) In current versions, the loops are over frame_out, that is with a 
+2 increment. Possibility to have a version with loops over frame_in
 (?? ask to the contact).

Which is the best ?

Contact : Stephanie.Even@enstb.org

Any publication of this application must acknowledge the transmedi@
project from the Image and Networks french research cluster.

To display the input video :
   mplayer Home_Trailer.yuv -demuxer rawvideo -rawvideo w=400:h=226

You need to have mplayer installed to be able to display the result.

For the sequential execution

  make seq : build the sequential program (named upscaling_seq)

  make run_seq : build first if needed, then run the sequential program

  make display_seq : only displays the video using mplayer

For the OpenMP parallel execution on multicores:

  make openmp : parallelize the code to OpenMP sources
  main.p4a.c, yuv.p4a.c and upscale.p4a.c and build the program 
  upscaling_openmp

  make run_openmp : build first if needed, then run the OpenMP parallel
  program

  make display_openmp : only displays the video using mplayer

For the CUDA parallel execution on nVidia GPU:

  make cuda : parallelize the code to CUDA source main.p4a.cu, 
  yuv.p4a.cu and upscale.p4a.cu and build the program upscaling_cuda

At that time :
   1) the variable nbframes was replaced by a #define NBFRAMES 3525 inside the
   main. Otherwise, the p4a_launcher_video_processing call 

   void p4a_launcher_video_processing(int nbframes, type_yuv_frame_in frame_in[nbframes], type_yuv_frame_out frame_out[nbframes]);

   is such it complains after nbframes at the compilation.

   Could be improved ? Is it possible to let the previous version with the 
   nbframes parameter ? It is a strong limitation.

   2) Now, not solved :
   In the upscale.p4a.c, in the upscale_chrominance procedure, the call to the
   p4a_launcher_upscale_chrominance(u, v, u, v, l, j, sizeChrW, sizeChrWout, frame_in, frame_out, jj, ll, u_fin, u_fout, v_fin, v_fout);
   creates the following complains at compilation with nvcc :
   
/home/even/par4all/examples/P4A/Upscaling/upscale.p4a.cu(155): error: identifier "u" is undefined

/home/even/par4all/examples/P4A/Upscaling/upscale.p4a.cu(155): error: identifier "v" is undefined

     In fact, I don't see where u and v are defined ???????
		
    3) Last, but not least, actually the message is

    /home/even/par4all/examples/P4A/Upscaling/main.p4a.cu(101): error: calling a host function from a __device__/__global__ function is not allowed		    corresponding to the call of the upscale procedure.	       
							      
make clean for cleaning.


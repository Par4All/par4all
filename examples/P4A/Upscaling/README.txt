This upscaling program was developed to test an architecture prototype
based on CPU and GPU and designed for multimedia transcoding and
processing inside core-network gateways. This work was achieved in the
TransMedi@ project framework from the French Images and Networks
research cluster. The upscaling function is the cubic one (6 points)
with a factor 2 (doubling the size of the image) taken from the H264
standard.  In takes a video in yuv format as input and displays the
upscaled video in yuv format as output.

Contact : Stephanie.Even@enstb.org

Any publication of this application must acknowledge the transmedi@
project from the Image and Networks french research cluster.

To display the input video :
   mplayer BUS_176x144_15_orig_01.yuv -demuxer rawvideo -rawvideo w=176:h=155

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

  make run_cuda

  make display_cuda

For the OpenCL parallel execution on GPU: make opencl, make run_opencl and make display_opencl.

In any case, make display will display the res.out output.

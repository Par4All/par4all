This upscaling program was developed to test an architecture prototype
based on CPU and GPU and designed for multimedia transcoding and
processing inside core-network gateways. This work was achieved in the
TransMedi@ project framework from the French Images and Networks
research cluster. The upscaling function is the cubic one (6 points)
with a factor 2 (doubling the size of the image) taken from the H264
standard.  In takes a video in yuv format as input and displays the
upscaled video in yuv format as output.


You need to have mplayer installed to be able to display the result.

For the sequential execution

  make seq : build the sequential program (named upscaling)

  make run_seq : build first if needed, then run the sequential program

  make display_seq : only displays the video using mplayer

(At that moment, only the sequential program works).

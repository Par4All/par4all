      subroutine desugar02(x, y, *, *)

C     Check desugaring of labelled DO loops

c     labelled loop
 400  do 500 i = 1, n
         print *, i
 500  continue

      end

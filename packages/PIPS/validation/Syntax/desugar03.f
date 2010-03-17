      subroutine desugar03(x, y, *, *)

C     Check desugaring of labelled DO loops with a forward reference

      read *, i
      
      if(i.lt.0) go to 400

c     labelled loop
 400  do 500 i = 1, n
         print *, i
 500  continue

      end

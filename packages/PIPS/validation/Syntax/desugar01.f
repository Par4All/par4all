      subroutine desugar01(x, y, *, *)

      read *, i, n

c     forward reference to 100
      if(i.lt.0) go to 100
c     forward reference to 200
      if(i.lt.0) go to 200
c     forward reference to 300
      if(i.lt.0) go to 300
c     forward reference to 400
      if(i.lt.0) go to 400

c     assign statement
      assign 200 to next

c     labelled logical if with alternate return
 100  if(x.gt.0.) return 2

c     labelled logical if with computed go to
 200  if(x.eq.0.) go to (100, 200, 300), i

c     labelled logical if with assigned go to
 300  if(x.lt.0.) go to next, (100, 200, 300)

c     labelled loop
      continue

 400  do 500 i = 1, n
         print *, i
 500  continue

      end

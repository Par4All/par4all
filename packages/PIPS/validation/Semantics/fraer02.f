      subroutine fraer02(x, y, q, r)

c     Example on Page 30, Fraer's PhD

c     We cannot get that r is decreasing because the transformer for the
c     while loop is computed withtout precondition information

c     We do not get r >= 0 at the print because the while loop
c     precondition computation probably does not execute a last
c     iteration under the proper conditions (as has been added for DO
c     loops)

      integer x, y, q, r

      if(x.lt.0.or.y.le.0) then
         stop
      else
         q = 0
         r = x
         do while(r.ge.y) 
            q = q + 1
            r = r - y
         enddo
      endif

      print *, x, y, q, r

      end

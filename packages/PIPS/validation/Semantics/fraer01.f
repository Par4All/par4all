      program fraer01

C     Example from Fraer's PhD, P. 72

C     We do get the convex approximation of the postcondition.

C     The redundancy elimination used in PIPS fails in this case.

C     To get a stronger postcondition by bounding x, y and z: see fraer03

      integer x, y, z

      read *, x, y, z

      if(x.lt.y) then
         max2 = y
         min2 = x
      else
         max2 = x
         min2 = y
      endif

      call printmin2(min2, x, y)
      call printmax2(max2, x, y)

      if(max2.lt.z) then
         max3 = z
         min3 = min 2
      else
         max3 = max2
         if(min2.gt.z) then
            min3 = z
         else
            min3 = min2
         endif
      endif

      call printmin3(min3, x, y, z)
      call printmax3(max3, x, y, z)

      print *, min2, max2, min3, max3

      end

      subroutine printmin2(min2, x, y)

      integer x, y

      print *, min2, x, y

      end

      subroutine printmax2(max2, x, y)

      integer x, y

      print *, max2, x, y

      end

      subroutine printmin3(min3, x, y, z)

      integer x, y, z

      print *, min3, x, y, z

      end

      subroutine printmax3(max3, x, y, z)

      integer x, y, z

      print *, max3, x, y, z

      end


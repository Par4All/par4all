! Check impact of spacing property

      integer function space02(x,y,z)
      integer x, y
      real*8 z

      if(z.gt.0.) then
         space02 = x+y
      else
         space02 = x-y
      endif
      return
      end

      program main
      integer j

      j = 2 + 3

      k = space01(1,j,3.)

      end


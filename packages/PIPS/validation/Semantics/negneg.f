C     Test of .NE. handling in the false branch

      subroutine negneg(i,j)

      if(i.ne.j) then
         i = j
      else
         k = 3
      endif

      print *, i, j, k

      end

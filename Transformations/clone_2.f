! trying clone transformations...
      program c2
      call clonee(1,1)
      call clonee(2,3)
      call clonee(2,2)
      end

      subroutine clonee(i, j)
      integer i, j
      if (i.eq.j) then
         print *, 'both arguments are equal'
      else
         print *, 'the arguments are different'
      endif
      print *, i, j
      end

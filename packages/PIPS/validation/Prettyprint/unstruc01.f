C     Check labelling by text_unstructured()

      subroutine unstruc01(x, x0)
      real x(10)

      do i = 1, 10
         if(x(i).gt.x0) goto 100
      enddo

      print *, "criterion not met"
      return
 100  print *, "criterion met"
      end

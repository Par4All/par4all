C     Check that the initial precondition is not used when the root
C     of the call graph is not a main program

      subroutine initial3
      print *, 'initial3'
      call pc
      end

      subroutine pc
      common /init/ i1, i2, i3, i4, i5
      data i1 /1/
      print *, 'pc'
      print *, i1, i2, i3, i4, i5
      end

      subroutine foo
      common /init/ i1, i2, i3, i4, i5
      data i2 /2/
      print *, 'foo'
      end

      block data 
      common /init/ i1, i2, i3, i4, i5
      data i3 /3/
      end

      block data bla
      common /init/ i1, i2, i3, i4, i5
      data i4 /4/
      end

      integer function boo(i)
      integer i
      common /init/ i1, i2, i3, i4, i5
      data i5 /5/
      BOO = i+i5
      end

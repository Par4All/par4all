      program initial2
      print *, 'initial'
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
      boo = i+i5
      boo = boo + 1
      end

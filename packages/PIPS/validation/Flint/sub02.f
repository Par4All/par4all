      program sub02
      external sub2
      integer sub2
      integer i
      call sub1
      i = sub2() + 1
      print *, i
      end

      integer function sub1()
      print *, 'hello from sub1'
      sub1 = 0
      end

      subroutine sub2
      print *, 'hello from sub2'
      end

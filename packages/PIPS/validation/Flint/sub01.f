      program sub
      print *, 'hello from sub'
      call sub1
      call sub2
      end

      subroutine sub1
      external sub2
      integer sub2
      print *, 'hello from sub1'
      print *, sub2()+1
      end

      integer function sub2()
      print *, 'hello from sub2'
      sub2 = 1
      return
      end

! ENTRIES...
      program entries
      print *, 'some entries'
      call fun1
      call bla2
      call foo4
      print *, 'done'
      call lee
      end
! LEE
      subroutine lee
      print *, 'lee entries'
      call bla3
      call foo1
      call fun
      print *, 'end'
      end
! FUN
      subroutine fun
      print *, 'sub fun'
      entry fun1
      print *, 'entry fun1 - return'
      return
      entry fun2
      print *, 'entry fun2'
      entry fun3
      print *, 'entry fun3'
      entry fun4
      print *, 'entry fun4'
      end
! BLA
      subroutine bla
      print *, 'sub bla'
      entry bla1
      print *, 'entry bla1'
      entry bla2
      print *, 'entry bla2 - return'
      return
      entry bla3
      print *, 'entry bla3'
      entry bla4
      print *, 'entry bla4'
      end
! FOO
      subroutine foo
      print *, 'sub foo'
      entry foo1
      print *, 'entry foo1'
      entry foo2
      print *, 'entry foo2'
      entry foo3
      print *, 'entry foo3'
      entry foo4
      print *, 'entry foo4 - return'
      end

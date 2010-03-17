      subroutine foo(*, i)
      print *, "foo is entered with ", i
      if(i.gt.0) then
         print *, 'RETURN 1 in FOO'
         return 1
      endif
      i = i + 1
      end

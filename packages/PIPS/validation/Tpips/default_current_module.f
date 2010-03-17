C     See which module is selected as current module when no main is
C     available: none, because several modules are candidate.

      subroutine default_current_module
      print *, 'coucou'
      end

      subroutine foo
      print *, 'foo'
      end

      subroutine bar
      print *, 'bar'
      end

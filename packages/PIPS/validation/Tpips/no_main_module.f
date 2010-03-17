C     Check that current_module_name is set even when no main is
C     defined, at least when the workspace contains only one module
C     
C     See default_module_name() for a case with no main module but with
C     several modules.

      subroutine no_main_module
      print *, 'coucou'
      end

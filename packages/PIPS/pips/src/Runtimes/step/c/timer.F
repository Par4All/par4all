#ifdef TEST_TIMER_F
      program test
      implicit none
      integer i
      INTEGER*8 CLOCK

      call step_timer_init(CLOCK)
      
      do i=1,100
         print *, i
      enddo
      call step_timer_event(CLOCK)
      do i=1,100
         print *, i
      enddo
      call step_timer_dump(CLOCK, "test_clock", 0)
      call step_timer_reset(CLOCK)
      do i=1,200
         print *, i
      enddo
     
      call step_timer_dump(CLOCK, "test_clock", 1)
      call step_timer_finalize(CLOCK);

      end
#endif

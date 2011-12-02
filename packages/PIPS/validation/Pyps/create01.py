from pyps import workspace
with workspace("dummy.c" ) as w:
    print "default behavior of abort_on_user_error", w.props.abort_on_user_error
with workspace("dummy.c", abort_on_user_error=True) as w:
    print "custom behavior of abort_on_user_error", w.props.abort_on_user_error

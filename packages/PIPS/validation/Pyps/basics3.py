from __future__ import with_statement # this is to work with python2.5
#!/usr/bin/env python
from pyps import *
workspace.delete("basics3")
with workspace("basics3.c",name="basics3",verbose=True,cppflags="-DI_LIKE_PIPS",
        ldflags="-Lthis_place_does_not_exist", recoverInclude=True,
        deleteOnClose=True) as ws:
    print ws.name
    print ws.dirname
    print ws.tmpdirname
    for i in ws : print i.name
    print ws["main"].name
    if "main" in ws : print "yes"
    else: print "no"
    ws.save()
    chk0=ws.checkpoint()
    ws.save()
    ws.fun.main.code="""
    int main() {
    puts("goodbye\\n");
    return 0;
    }
    """
    b=ws.compile(Maker(),os.path.join(ws.tmpdirname,"supertmp"),
            "b.out", "all", CFLAGS="-O0 -g")
    ws.run(b,["12"])
    ws.fun.main.display()
    ws.save()
    ws.restore(chk0)
    ws.save()
    ws.fun.main.display()
    c=ws.compile(Maker(),os.path.join(ws.tmpdirname,"supertmp"),
            "c.out", "all", CFLAGS="-O0 -g")
    ws.run(c,["12"])
    ws.activate(module.print_code_regions)
    ws.activate("print_code_regions")


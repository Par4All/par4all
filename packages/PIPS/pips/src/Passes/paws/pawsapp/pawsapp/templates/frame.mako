<!DOCTYPE html public "-//W3C//DTD HTML 4.0 Transitional//EN" "http://www.w3.org/TR/REC-html40/loose.dtd">
<html>
	<head>
		${self.head()}
	</head>
        <body style="padding: 0; position: absolute; top: 0; left: 50px; right: 50px; bottom: 0; margin: 0">
                <div class="header" style="margin: 0; padding: 0; position: relative">
                        <table><tr><td>
                                <a href="/paas/index"><img src="/paws.jpg" height="50" style="border: none"/></a>
                        </td><td width="10%">
                                <br/>
                        </td><td>
                                <p><h2>${self.header()}</h2></p>
                        </td></tr></table>
                </div>
                <div class="site-content" style="top: 0">
                        ${self.content()}
                </div>
                <div class="footer">
                        <p align="left"><a href="http://pips4u.org" target="_blank"><img src="/pips4u.gif" height="30px" style="border: none"/></a> INSIDE!</p>
                        <p align="right">© 2011  MINES ParisTech</p>
                </div>
        </body>
</html>

<%doc>
  Main template
</%doc>

## Page title
<%def name="pagetitle()">
PAWS
</%def>

## Title
<%def name="title()">
PAWS
</%def>

## Custom CSS
<%def name="js_slot()">
</%def>

## Custom Javascript
<%def name="css_slot()">
</%def>


<!DOCTYPE HTML PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">

<html>
  <head>
 
   <title>${self.pagetitle()}</title>
    
    ## Stylesheets
    ${h.stylesheet_link(url("/css/jq/cupertino/jquery-ui-1.8.10.custom.css"), media="all")}
    ##${h.stylesheet_link(url("/css/jq/jquery-linedtextarea.css"), media="all")}
    ##${h.stylesheet_link(url("/css/jq/jquery.jqzoom.css"), media="all")}
    ##${h.stylesheet_link(url("/css/pygments.css"), media="all")}
    ${h.stylesheet_link(url("/css/normal.css"), media="all")}
    ${h.stylesheet_link(url("/css/print.css"),  media="print")}
    ${self.css_slot()}

    ## Javascript
    ${h.javascript_link(url("/jq/jquery-1.4.4.min.js"))}
    ${h.javascript_link(url("/jq/jquery-ui-1.8.10.custom.min.js"))}
    ##${h.javascript_link(url("/jq/jquery-linedtextarea.js"))}
    ##${h.javascript_link(url("/jq/jquery.jqzoom-core.js"))}
    ${h.javascript_link(url("/js/base.js"))}
    ${self.js_slot()}
  </head>

  <body>

    ## Header
    <div id="header">
      <a href="/">${h.image("/img/paws-small.png", "PAWS Logo")}</a>
      <h1>${self.title()}</h1>
    </div>

    ## Main content
    <div id="site-content">
      ${self.content()}
    </div>

    ## Footer
    <div id="footer">
      <p><a href="http://pips4u.org" target="_blank">
	${h.image("/img/pips-small.png", "PIPS4u logo")}</a> INSIDE!</p>
      <p id="copyright">© 2011-2012 MINES ParisTech</p>
    </div>

  </body>
</html>

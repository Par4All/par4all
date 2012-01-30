<%doc>
  Generic page for a tool
</%doc>


<%inherit file="base.mako"/>

<%namespace name="w"   file="pawsapp:templates/lib/widgets.mako"/>

<%def name="css_slot()">
${h.stylesheet_link(request.static_url("pawsapp:static/css/jq/jquery-linedtextarea-min.css"), media="all")}
${h.stylesheet_link(request.static_url("pawsapp:static/css/pygments-min.css"), media="all")}
</%def>

<%def name="js_slot()">
${h.javascript_link(request.route_url("routes.js"))}
${h.javascript_link(request.static_url("pawsapp:static/jq/jquery-linedtextarea-min.js"))}
<script type="text/javascript">
  operation = "demo";
</script>
${h.javascript_link(request.static_url("pawsapp:static/js/tutorial.js"))}
</%def>


## ONE COLUMN

<%def name="content()">

<div class="container">

  <div class="hero-unit" style="padding:.5em 1em">
    <h2>${h.image(request.static_url("pawsapp:static/img/favicon-trans.gif"), u"PAWS icon")}
      ${name}
    </h2>
  </div>

  <table>
    <tr valign="top">
      <td>
        <div id="demo">
	  <br/>
	  <label for="step" class="table_header">Current demo step:&nbsp;&nbsp; </label>
	  <input type="text" id="step" class="ui-widget ui-widget-content slider_label"/><br/>
	  <label for="all_steps" class="table_header">Number of all the steps:&nbsp;&nbsp; </label>
	  <input type="text" id="all_steps" class="ui-widget ui-widget-content slider_label"/>
	  <div id="demo_slider"></div>
	  <br/>
	  <table>
	    <tr>
	      <td>
		<div class='table_header'>SOURCE:</div>
	      </td>
	      <td>
		<div class='table_header'>SCRIPT:</div>
	      </td>
	    </tr>
	    <tr>
	      <td valign='top' align='left' width='700px'>
		<div id="source">
		  <textarea name="sourcecode" id="sourcecode" class="output" rows="33" cols="85"
			    onkeydown="handle_key_down(this, event)">${source}</textarea>
		</div>
	      </td>
	      <td valign='top' align='left' width='700px'>
		<div id="sourcetpips" class="output">
		  <p align="center">${tpips|n}</p>
		</div>
	      </td>
	    </tr>
	  </table>
        </div>
      </td>
    </tr>
  </table>

</div>
</%def>

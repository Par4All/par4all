<%doc>
  Generic page for a tool
</%doc>


<%inherit file="base.mako"/>

<%def name="title()">
${descr}
% if advanced:
(advanced)
% endif
</%def>

<%def name="css_slot()">
${h.stylesheet_link(request.static_url("pawsapp:static/css/jq/jquery-linedtextarea.css"), media="all")}
${h.stylesheet_link(request.static_url("pawsapp:static/css/jq/jquery.jqzoom.css"), media="all")}
${h.stylesheet_link(request.static_url("pawsapp:static/css/pygments.css"), media="all")}
${h.stylesheet_link(request.static_url("pawsapp:static/css/tool.css"), media="all")}
</%def>

<%def name="js_slot()">
${h.javascript_link(request.static_url("pawsapp:static/jq/jquery-linedtextarea.js"))}
${h.javascript_link(request.static_url("pawsapp:static/jq/jquery.jqzoom-core.js"))}
${h.javascript_link(request.static_url("pawsapp:static/js/tool.js"))}
<script type="text/javascript">
  operation = "${tool}";
</script>
</%def>


## Page content

<iframe id="iframetoprint" style="height: 0px; width: 0px; position: absolute; -moz-opacity: 0; opacity: 0"></iframe>

<table class="ui-widget ui-widget-content ui-corner-all">
  <tr valign="top">

    <td id="left">

      <div id="resizing_source" style="text-align:right">
	${h.submit("Aplus", value="A+")}
	${h.submit("Aminus", value="A-")}
      </div>

      <h3>TYPE OR SELECT SOURCE CODE FROM:</h3>

      <p><b>A set of classic examples:</b></p>
      ${h.submit("classic-examples-button", u"BROWSE")}
      
      <p><b>Or from your own test cases:</b></p>
      <input type="submit" value="BROWSE" id="pseudobutton"/>
      <input type="file" id="inp" name="file" class="hide"/>
      <input type="text" id="pseudotextfile" readonly="readonly"/>

      <p>${h.submit("run-button", value="RUN")}</p>

      <div class="save_results">
	<p><a href="/res/result_file" id="save_button_link" style="text-decoration: none"><input value="SAVE RESULT" type="submit" id="save_button"/></a></p>
      </div>

      <div class="print_results">
	<p><input value="PRINT RESULT" type="submit" id="print_button"/></p>
      </div>

      <div>
	% if advanced:
	${h.link_to(u"basic mode", url=request.route_url("tool_basic", tool=tool))}
	% else:
	${h.link_to(u"advanced mode", url=request.route_url("tool_advanced", tool=tool))}
	% endif
      </div>

    </td>

    <td id="right">

      <div id="tabs">

	## Tab headers
	<ul>
	  <li><a href="#tabs-1" id="source_tab_link1">SOURCE</a></li>
	  <li><a href="#result" id="result_tab_link">${tool.upper()}</a></li>
	  <li><a href="#graph">GRAPH</a></li>
	</ul>

	## Source code tab
	<div id="tabs-1">
	  <form name="language1">
	    <label for="lang1">Language: </label>
	    <input name="lang1" value="not yet detected." readonly="readonly"/>
	  </form>
	  <form name="source1">
	    <textarea name="sourcecode1" id="sourcecode1" rows="27" cols="120" onkeydown="handle_key_down(this, event)">Put your source code here.</textarea>
	  </form>
	</div>

	## Result tab
	<div id="result">
	  <div id="multiple-functions">
	  </div>
	  <div id="resultcode">
	    Result of the transformation will be displayed here.
	  </div>
	</div>

	## Graph tab
	<div id="graph">
	  <div id="dependence_graph">
	    Please wait, it might take long time.
	  </div>
	</div>

      </div>

    </td>

  </tr>
</table>


## Dialog boxes

## Classic examples dialog
<div id="classic-examples-dialog" title="Please select an example">
    % for ex in examples:
    ${h.submit(ex, ex)}<br/>
    % endfor
</div>	

<div id="dialog-choose-function" title="Select function to transform.">
  <div class="choose-function" id="choose-function-buttons"></div>
</div>


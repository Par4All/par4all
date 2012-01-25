<%doc>
  Generic page for a tool
</%doc>


<%inherit file="base.mako"/>

<%namespace name="adv" file="pawsapp:templates/lib/advanced.mako"/>

<%def name="css_slot()">
${h.stylesheet_link(request.static_url("pawsapp:static/css/jq/jquery-linedtextarea-min.css"), media="all")}
${h.stylesheet_link(request.static_url("pawsapp:static/css/jq/jquery.jqzoom-min.css"), media="all")}
${h.stylesheet_link(request.static_url("pawsapp:static/css/pygments-min.css"), media="all")}
</%def>

<%def name="js_slot()">
${h.javascript_link(request.route_url("routes.js"))}
${h.javascript_link(request.static_url("pawsapp:static/jq/bootstrap-tabs-min.js"))}
${h.javascript_link(request.static_url("pawsapp:static/jq/bootstrap-modal-min.js"))}
${h.javascript_link(request.static_url("pawsapp:static/jq/jquery-linedtextarea-min.js"))}
${h.javascript_link(request.static_url("pawsapp:static/jq/jquery.jqzoom-core-pack-min.js"))}
<script type="text/javascript">
  operation = "${tool}";
</script>
${h.javascript_link(request.static_url("pawsapp:static/js/init.js"))}
</%def>


## LEFT COLUMN

<%def name="left_column()">

<div id="resizing_source" style="text-align:right">
  ${h.link_to(u"A+", id="aplus", class_="btn small")}
  ${h.link_to(u"A-", id="aminus", class_="btn small")}
</div>

<h4>Type or select source code from:</h4>

<dl>
  <dt>A set of classic examples:</dt>
  <dd>
    <a class="btn small" data-controls-modal="classic-examples-dialog" data-backdrop="static">BROWSE</a>
    <br/><br/>
  </dd>
  <dt>Or from your own test cases:</dt>
  <dd>
    <input type="submit" value="BROWSE" id="pseudobutton"/>
    <input type="file" id="inp" name="file" class="hide"/>
    <input type="text" id="pseudotextfile" readonly="readonly" class="span3"/>
  </dd>
</dl>

## Properties (for advanced mode)
% if advanced:
<form class="form-stacked">
  ${adv.properties_fields(props)}
  ${adv.analyses_fields(analyses)}
  ${adv.phases_fields(phases)}
</form>
% endif


<p>${h.link_to(u"RUN", id="run-button", class_="btn small")}</p>
<p>${h.link_to(u"SAVE RESULT",  id="save-button",  class_="btn small disabled")}</p>
<p>${h.link_to(u"PRINT RESULT", id="print-button", class_="btn small disabled")}</p>

<div>
  % if advanced:
  ${h.link_to(u"Basic mode", url=request.route_url("tool_basic", tool=tool))}
  % else:
  ${h.link_to(u"Advanced mode", url=request.route_url("tool_advanced", tool=tool))}
  % endif
</div>

</%def>


## MAIN COLUMN

<%def name="main_column()">

<iframe id="iframetoprint" style="height: 0px; width: 0px; position: absolute; -moz-opacity: 0; opacity: 0"></iframe>

<div class="hero-unit" style="padding:.5em 1em">
  <h2>${descr}
    % if advanced:
    <span class="label important">advanced</span>
    % endif
  </h2>
</div>

<div id="op-tabs">

  ## Tab headers
  <ul class="tabs" data-tabs="tabs">
    <li class="active"><a href="#tabs-1" id="source_tab_link1">SOURCE</a></li>
    <li><a href="#result" id="result_tab_link">${tool.upper()}</a></li>
    <li><a href="#graph"  id="graph_tab_link">GRAPH</a></li>
  </ul>
  
  <div class="tab-content">

    ## Source code panel
    <div id="tabs-1" class="active tab-pane">
      <form>
	<fieldset style="padding-top:0">
	  <label for="lang1">Language </label>
	  <div class="input">
	    <input id="lang1" value="not yet detected." readonly="readonly"/>
	  </div>
	</fieldset>
      <textarea id="sourcecode1" class="span16" rows="27" onkeydown="handle_keydown(this, event)">Put your source code here.</textarea>
      </form>
    </div>

    ## Result panel
    <div id="result" class="tab-pane">
      <div id="multiple-functions">
      </div>
      <div id="resultcode" class="span16">
	Result of the transformation will be displayed here.
      </div>
    </div>

    ## Graph panel
    <div id="graph" class="tab-pane">
    </div>

  </div>
</div>
</%def>


## DIALOG BOXES

<%def name="dialogs()">

## Classic examples dialog

<div class="modal hide fade" id="classic-examples-dialog" style="display: none;">
  <div class="modal-header">
    <a class="close" href="#">×</a>
    <h3>Please select an example</h3>
  </div>
  <div class="modal-body">
    % for ex in examples:
    ${h.link_to(ex, id=ex, class_="btn small span8")}<br/>
    % endfor
  </div>
</div>

## Functions dialog

<div class="modal hide fade" id="choose-function-dialog" style="display: none;">
  <div class="modal-header">
    <a class="close" href="#">×</a>
    <h3>Select function to transform</h3>
  </div>
  <div class="modal-body" >
  </div>
</div>

</%def>

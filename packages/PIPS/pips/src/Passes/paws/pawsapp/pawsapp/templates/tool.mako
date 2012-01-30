<%doc>
  Generic page for a tool
</%doc>


<%inherit file="base.mako"/>

<%namespace name="w"   file="pawsapp:templates/lib/widgets.mako"/>
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
    <button class="btn small" id="aminus">A-</button>
    <button class="btn small" id="aplus">A+</button>
  </div>

  <h4>Type or select source code from:</h4>

  <form target="upload_target" action="${request.route_url('upload_user_file')}"
	enctype="multipart/form-data" method="post" id="upload_form">

    <div class="clearfix">
      <button class="btn primary span3" id="classic-button"
	      data-controls-modal="classic-examples-dialog"
	      data-backdrop="static">Classic examples »</button>
    </div>

    <div class="clearfix">
      <p>or from your own test cases:</p>
      <button id="pseudobutton" class="btn primary span3">Browse »</button>
      <input type="file" id="upload_input" name="file" class="inp-hide"/>
      <input type="text" id="pseudotextfile" readonly="readonly" class="span3"/>
    </div>

  </form>

  <div class="clearfix">
    <p>
      <button class="btn primary disabled span3" id="run-button">Run »</button>
    </p>
    <p>
      <button class="btn disabled span3" id="save-button">Save Result</button><br/>
      <button class="btn disabled span3" id="print-button">Print Result</button>
    </p>
  </div>

  % if advanced:

  <h4>Advanced mode</h4>

  <div class="clearfix">
    <button class="btn success span3" data-keyboard="true" data-backdrop="static"
	    data-controls-modal="adv-props-modal">Properties</button><br/>
    <button class="btn success span3" data-keyboard="true" data-backdrop="static"
	    data-controls-modal="adv-analyses-modal">Select Analyses</button><br/>
    <button class="btn success span3" data-keyboard="true" data-backdrop="static"
	    data-controls-modal="adv-phases-modal">Phases</button>
  </div>
  
  ## "Properties" modal
  ${w.modal(u"Properties", advprop_body, id="adv-props-modal")}
  <%def name="advprop_body()">
  ${adv.properties_fields(props)}
  </%def>

  ## "Select Analyses" modal
  ${w.modal(u"Select Analyses", advanl_body, id="adv-analyses-modal")}
  <%def name="advanl_body()">
  ${adv.analyses_fields(analyses)}
  </%def>

  ## "Phases" modal
  ${w.modal(u"Phases", advphases_body, id="adv-phases-modal")}
  <%def name="advphases_body()">
  ${adv.phases_fields(phases)}
  </%def>

  % endif

  <div class="clearfix" style="text-align: right">
    Switch to 
    % if advanced:
    ${h.link_to(u"basic mode", url=request.route_url("tool_basic", tool=tool))}
    % else:
    ${h.link_to(u"advanced mode", url=request.route_url("tool_advanced", tool=tool))}
    % endif
    »
  </div>

</%def>


## MAIN COLUMN

<%def name="main_column()">

<iframe id="iframetoprint" style="height: 0px; width: 0px; position: absolute; -moz-opacity: 0; opacity: 0"></iframe>

<div class="hero-unit" style="padding:.5em 1em">
  <h2>${h.image(request.static_url("pawsapp:static/img/favicon-trans.gif"), u"PAWS icon")}
    ${descr}
    % if advanced:
    <span class="label important">advanced</span>
    % endif
  </h2>
</div>

<div id="op-tabs">

  ## Tab headers
  <ul class="tabs" data-tabs="tabs">
    ${w.source_tab(id="1", active=True)}
    <li id="result_tab">${h.link_to(tool.upper(), url="#result")}</li>
    <li id="graph_tab">${h.link_to(u"GRAPH", url="#graph")}</li>
  </ul>
  
  <div class="tab-content">

    ## Source code panel
    ${w.source_panel(id="1", active=True)}

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

## Invisible iframe for file upload
<iframe id="upload_target" name="upload_target" class="hide"></iframe> 

<div id="source_panel_skel" class="hide">
  ${w.source_panel(id="__skel__")}
</div>

</%def>


## DIALOG BOXES

<%def name="dialogs()">

## Classic examples dialog

${w.modal(u"Please select an example", classic_body, "classic-examples-dialog")}

<%def name="classic_body()">
% for ex in examples:
${h.link_to(ex, id=ex, class_="btn small span8")}<br/>
% endfor
</%def>


## Functions dialog

${w.modal(u"Select function to transform", functions_body, "choose-function-dialog")}

<%def name="functions_body()">
</%def>


</%def>

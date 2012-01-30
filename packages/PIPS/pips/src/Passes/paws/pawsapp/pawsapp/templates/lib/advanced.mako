<%doc>
  Widgets for advanced mode
</%doc>


## PROPERTIES (advanced mode)

<%def name="properties_fields(props)">

<fieldset>

  <h4>Properties</h4>

  ## BOOL properties
  % if "bool" in props:
  <div class="clearfix">
    <label id="bools">True/False</label>
    <div class="input">
      <ul class="inputs-list">
	% for p in props["bool"]:
        <li>
          <label>
	    ${h.checkbox("bools", value=p["name"], checked=p["val"])}
	    <span>${p["name"]}</span>
          </label>
        </li>
	% endfor
      </ul>
    </div>
  </div>
  % endif

  ## INT properties
  % if "int" in props:
  <div class="clearfix">
    <label id="ints">Integer</label>
    <div class="input">
      <ul class="inputs-list">
	% for p in props["int"]:
        <li>
          <label>
	    ${h.checkbox("bools", value=p["name"], checked=True)}
	    <span>${p["name"]}</span>
          </label>
	  ${h.text(p["name"], value=p["val"], size=5)}
        </li>
	% endfor
      </ul>
    </div>
  </div>
  % endif

  ## STR properties
  % if "str" in props:
  <div class="clearfix">
    <label id="strs">String</label>
    <div class="input">
      <ul class="inputs-list">
	% for p in props["str"]:
        <li>
          <label>
	    ${h.checkbox("bools", value=p["name"], checked=True)}
	    <span>${p["name"]}</span>
          </label>
	  ${h.select(p["name"], "", [(v, v) for v in p["val"]])}
        </li>
	% endfor
      </ul>
    </div>
  </div>
  % endif

</fieldset>
</%def>


## ANALYSES (advanced mode)

<%def name="analyses_fields(analyses)">

<h4>Select analyses</h4>

<form class="form-stacked">
  <fieldset>
    <div class="clearfix">
      <div class="input">
        <ul class="inputs-list">
	  % for a in analyses:
          <li>
            <label>
	      ${h.checkbox("analyses", value=a, checked=True)}
	      <span>${a}</span>
            </label>
	    ${h.select(a, "", [(v["name"], v["name"]) for v in analyses[a]])}
          </li>
	  % endfor
        </ul>
      </div>
    </div>
  </fieldset>
</form>

</%def>


## PHASES (advanced mode)

<%def name="phases_fields(phases)">

<h4>Phases</h4>

<fieldset>
  <div class="clearfix">
    <div class="input">
      <ul class="inputs-list">
	% for p in phases["PHASES"]:
        <li>
          <label>
	    ${h.checkbox("phases", value=p["name"], checked=True)}
	    <span>${p["name"]}</span>
          </label>
        </li>
	% endfor
      </ul>
    </div>
  </div>
</fieldset>

</%def>

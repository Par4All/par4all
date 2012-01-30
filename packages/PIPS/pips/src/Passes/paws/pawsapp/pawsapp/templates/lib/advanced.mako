<%doc>
  Widgets for advanced mode
</%doc>


## PROPERTIES (advanced mode)

<%def name="properties_fields(props)">

<table>

  ## BOOL properties
  % if "bool" in props:
  <tr>
    <td style="border-top-style: none"><span class="label success">True/False</span></td>
    <td style="border-top-style: none">
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
    </td>
  </tr>
  % endif

  ## INT properties
  % if "int" in props:
  <tr>
    <td><span class="label success">Integer</span></td>
    <td>
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
    </td>
  </tr>
  % endif

  ## STR properties
  % if "str" in props:
  <tr>
    <td><span class="label success">String</span></td>
    <td>
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
    </td>
  </tr>
  % endif
  
</table>
</%def>


## ANALYSES (advanced mode)

<%def name="analyses_fields(analyses)">

<div class="input">
  <ul class="inputs-list">
    % for a in analyses:
    <li class="clearfix">
      <label>
	${h.checkbox("analyses", value=a, checked=True)}
	<span>${a}</span>
      </label>
      ${h.select(a, "", [(v["name"], v["name"]) for v in analyses[a]])}
    </li>
    % endfor
  </ul>
</div>

</%def>


## PHASES (advanced mode)

<%def name="phases_fields(phases)">

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

</%def>

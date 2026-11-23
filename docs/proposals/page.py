"""HTML for the reach proposals report. `render()` is called by build_proposals.py with real numbers."""

from __future__ import annotations

import html


def esc(s) -> str:
    return html.escape(str(s))


def fmt(n) -> str:
    return f"{n:,}"


def prune(criteria, variants, pick):
    """criteria: [(name, weight %, why)], variants: [(name, [scores 1-5])] -> table HTML, frozen before the build."""
    head = "".join(f"<th class='num' title='{esc(why)}'>{esc(n)}<br><span class='muted'>{w} %</span></th>" for n, w, why in criteria)
    rows = []
    for name, scores in variants:
        total = sum(s * w for s, (_, w, _) in zip(scores, criteria)) / 100
        cls = " class='pick'" if name == pick else ""
        cells = "".join(f"<td class='num'>{s}</td>" for s in scores)
        rows.append(f"<tr{cls}><td>{esc(name)}{' <span class=badge>pick</span>' if name == pick else ''}</td>{cells}<td class='num'><b>{total:.2f}</b></td></tr>")
    why = "".join(f"<li><b>{esc(n)}</b> ({w} %): {esc(y)}</li>" for n, w, y in criteria)
    return (f"<details><summary class='muted small'>How the pick was made: criteria fixed before the variants were built, scores 1–5</summary>"
            f"<ul class='small'>{why}</ul></details><table><tr><th>Variant</th>{head}<th class='num'>Weighted</th></tr>{''.join(rows)}</table>")


def card(title, thesis, body, best, loses, picked=False, extra_cls=""):
    return (f"<div class='card{' picked' if picked else ''} {extra_cls}'><h3>{title}{' <span class=badge>pick</span>' if picked else ''}</h3>"
            f"<div class='muted small'>{thesis}</div>{body}"
            f"<div class='trade'><b>Best when</b><span>{best}</span><b>Loses when</b><span>{loses}</span></div></div>")


def render(f, B, blob, mask_blob, css, js) -> str:
    c = f["checks"]
    lv_line = lambda k: f"reachable {fmt(sum(k[x] for x in (3, 4, 5, 6)))} · 4-DOF {fmt(k[4] + k[5] + k[6])} · hemisphere {fmt(k[5] + k[6])} · dexterous {fmt(k[6])}"  # noqa: E731
    cc = f["cloud_counts"]
    box = f["box"]
    cfg = f["config"]
    met = f["metrics"]

    # ---------------------------------------------------------------- D scores derived from the measurements
    def band(x, cuts):
        for score, cut in cuts:
            if x >= cut:
                return score
        return 1

    relearn = {"traffic": 5, "viridis": 4, "rainbow": 4, "thermal": 3, "okabe": 2, "today": 3}
    d_names = {"traffic": "D1 Traffic light", "viridis": "D2 Viridis, light = least", "rainbow": "D3 Rainbow bands", "thermal": "D4 Thermal", "okabe": "D5 Named colours", "today": "Today (reference)"}
    d_variants = []
    for k in ("traffic", "viridis", "rainbow", "thermal", "okabe", "today"):
        m = met[k]
        cvd = min(m["deuteranopia"], m["protanopia"], m["tritanopia"])
        d_variants.append((d_names[k], [band(cvd, [(5, 30), (4, 22), (3, 15), (2, 10)]), 5 if m["lightness_order"] else 1,
                                        band(m["vs_problem"], [(5, 50), (4, 35), (3, 25), (2, 15)]), band(m["vs_white"], [(5, 60), (4, 45), (3, 35), (2, 25)]), relearn[k]]))
    d_crit = [("Colour-blind safe", 30, "smallest colour difference (CIE76 ΔE) between the four levels under deuteranopia, protanopia and tritanopia (Machado 2009); ≥30 → 5, ≥22 → 4, ≥15 → 3, ≥10 → 2"),
              ("More = one direction", 25, "lightness changes the same way from 3 to 6, so order reads without the legend (and survives greyscale)"),
              ("Not a problem colour", 20, "ΔE from the collision red and singular purple the viewer already uses; ≥50 → 5, ≥35 → 4, ≥25 → 3"),
              ("Visible on white", 10, "ΔE of the faintest level from the white background"),
              ("Nothing to relearn", 15, "judgement: how familiar the convention already is")]
    d_best = max((v for v in d_variants if not v[0].startswith("Today")), key=lambda v: sum(s * w for s, (_, w, _) in zip(v[1], d_crit)))[0]
    mt_rows = "".join(
        f"<tr{' class=pick' if d_names[k] == d_best else ''}><td>{d_names[k]}</td>"
        + "".join(f"<td class='num'>{met[k][x]:.0f}</td>" for x in ("normal", "deuteranopia", "protanopia", "tritanopia"))
        + f"<td>{'yes' if met[k]['lightness_order'] else 'no'}</td><td class='num'>{met[k]['vs_problem']:.0f}</td><td class='num'>{met[k]['vs_white']:.0f}</td>"
        + "<td>" + " ".join(f"<span class='sw' style='background:rgb{tuple(col)}'></span>" for col in f_schemes(k)) + "</td></tr>"
        for k in ("traffic", "viridis", "rainbow", "thermal", "okabe", "today"))

    rf = f["range_facts"]
    rf_text = " · ".join(f"{k.split(' ')[0]} {v[0]:+.0f}…{v[1]:+.0f}°" for k, v in rf.items())
    cons = f["cons_counts"]

    sel = f["sel_pos"]
    picks = [
        ("A", "Choose IK solution types", "A2 Eight posture tiles + the <code>IKBranches</code> type", "Any of the 255 combinations, and each tile shows the arm in that posture, so “elbow up” needs no words; the type generates it for any arm."),
        ("B", "Joint ranges", "B2 One histogram per joint", "Bar height = how many working solutions use that angle; the limit is the grey band behind; gaps show (a box plot hides them)."),
        ("C", "Tiltable Ask-here box", "C1 Drag to tilt, numbers mirror it", "Viser's gizmo already rotates (the viewer turns it off today); the approach spikes are drawn inside the box and the angle set is spelled out under it."),
        ("D", "Colour scheme", f"{d_best}", "Best measured: colour-blind safe, darker-to-lighter in one direction, far from the red/purple that mean collision/singular; compas_fab's colours, run light (least) to dark (most) because the least level is the outer shell."),
        ("E", "Histogram legend + selector", "E1 Clickable bar legend", "The histogram IS the legend and the Show control: click any bar to hide it; Show follows Colour by; “unreachable” is a bar, which gives your “All”."),
        ("F", "Wizard in the Workspace panel", "F2 Used-settings card", "The fields show what the result used; editing one lights Compute workspace, hides the old map and offers Reset — exactly your flow, no new screens."),
        ("G", "Downstream API", "<a href='#g2'>G′ P1</a> attribute namespace (round 2; replaces G2)", "<code>ws.dexterous.sample()</code>, <code>ws.hemisphere.inside(belt).only(elbow=\"up\").sample_pairs()</code>: your words, IK configurations instead of branches, one object for every kind, a <code>Cell</code> for several chains. All five round-2 shapes ran on the UR5e."),
        ("H", "What the FK cloud is", "Answered below", "It is the box-maker: 20,000 random joint vectors → FK; their bounding box is the voxel grid. Colours = the joint check of each sample."),
    ]
    pick_rows = "".join(f"<tr><td><a href='#{k.lower()}'>{k}</a></td><td>{a}</td><td><b>{p}</b></td><td>{w}</td></tr>" for k, a, p, w in picks)

    vpanel_ws = lambda extra: f"<div class='vpanel'><div class='vfold'>Workspace</div>{extra}</div>"  # noqa: E731
    e_select = ("<div class='vrow'><label>Colour by</label><select><option>pose DOF</option><option>IK solutions</option>"
                "<option>fraction of orientations</option><option value='consistency'>FK/IK consistency (demo)</option></select></div>")

    e_cards = [
        ("bars", "E1 Clickable bar legend", "The histogram is the legend: one coloured bar per Colour-by bin, height = count; click a bar to show or hide those points.",
         "you want the count and the colour key in one glance, and any mix of bins", "a continuous colour (fraction) has 10+ bins: bars get thin (still fine at 11)", True),
        ("strip", "E2 Stacked strip", "One 100 % bar split by count, the labels underneath; click a segment or a label.",
         "the panel is narrow and proportions matter more than counts", "a bin is tiny (here 128 hemisphere points next to 4,904 empty): its segment is a sliver", False),
        ("range", "E3 Histogram + range slider", "The histogram above a two-handle slider: show the bins between the handles.",
         "the colour is an amount (fraction, IK solutions) and you want “between 3 and 6”", "you want two bins that are not next to each other (e.g. 3 and 6 only)", False),
        ("list", "E4 Legend list", "One row per bin: checkbox, swatch, full label, a count bar and the number.",
         "labels are long or a screen reader matters", "it is the tallest (9–11 rows for IK solutions / fraction)", False),
        ("atleast", "E5 “At least” threshold", "One handle: show every point at or above a bin; bars are cumulative, like today's Show (dexterous ⊂ hemisphere ⊂ 4-DOF ⊂ reachable).",
         "you think in nested workspaces", "you want exactly one band, e.g. only the 4-DOF ring that is not hemisphere", False),
    ]
    e_html = "".join(card(t, th, vpanel_ws(e_select + "<div class='hctl'></div>") + "<canvas></canvas><div class='readout'></div>", b, l, pk, f"evar' data-kind='{k}")
                     for k, t, th, b, l, pk in e_cards)

    used_mm = round(f["step_mm"])
    fields = lambda: (  # noqa: E731
        f"<div class='vrow'><label>Voxel size XYZ (mm)</label><div class='trio'><input type='number' data-k='vx' value='{used_mm}' step='5' min='20'>"
        f"<input type='number' data-k='vy' value='{used_mm}' step='5' min='20'><input type='number' data-k='vz' value='{used_mm}' step='5' min='20'></div></div>"
        "<div class='vrow'><label>Orientation step (°)</label><input type='number' data-k='tilt' value='45' step='5' min='5' max='90'></div>"
        "<div class='vrow'><label>Roll step (°)</label><input type='number' data-k='roll' value='60' step='5' min='5' max='180'></div>")
    btns = "<div class='vrow'><span></span><div class='pills'><button class='btn compute'>Compute workspace</button><button class='btn reset'>Reset</button></div></div>"
    f1 = (f"<div class='vpanel'><div class='vfold'>Workspace</div><div class='steps'>"
          f"<div class='step done'><span class='dot'>1</span><div><b>Robot</b> <span class='muted'>ur5e · 6 DOF · closed-form IK</span></div></div>"
          f"<div class='step done'><span class='dot'>2</span><div><b>Where to look</b> <span class='muted'>automatic: FK of {fmt(f['n_fk'])} random joint vectors → box "
          f"{box['size'][0]:.2f} × {box['size'][1]:.2f} × {box['size'][2]:.2f} m</span></div></div>"
          f"<div class='step'><span class='dot'>3</span><div><b>Resolution</b>{fields()}</div></div>"
          f"<div class='step'><span class='dot'>4</span><div><b>Run</b>{btns}<div class='cost'></div></div></div></div><div class='status'></div></div>")
    f2 = (f"<div class='vpanel'><div class='vfold'>Workspace</div><div class='muted small'>Computed with — edit to change</div>"
          f"{fields()}<div class='small muted'><span data-was='vx'></span> <span data-was='tilt'></span> <span data-was='roll'></span></div>{btns}<div class='cost'></div><div class='status'></div></div>")
    f3 = (f"<div class='vpanel'><div class='vfold'>Workspace</div><div class='recipe'></div><div class='pills'><button class='pill edit'>Edit settings…</button></div><div class='drawer'>{fields()}{btns}<div class='cost'></div></div><div class='status'></div></div>")

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reach Viewer Proposals</title><style>{css}</style></head><body><main>
<h1>BAM Reach v2 — the viewer, eight asks, 27 variants</h1>
<div class="muted">robot_reach · 2026-11-23 · every point cloud on this page is drawn live from the saved UR5e result
(<code>out/ur5e</code>: {fmt(f['P'])} grid points × {f['O']} orientations × {f['S']} IK branches → {fmt(f['n_sol'])} working solutions). Drag any picture to turn it.</div>
<div class="verdict"><b>One pick per ask below; nothing in <code>src/</code> changed.</b> These are proposals for the viewer agent to build.
The two prototypes that are code (the IK-type message for A, the workspace API for G) run on the real UR5e result in <code>docs/proposals/</code>.
Biggest surprise: the closed-form solver's elbow slot means “elbow up” on one shoulder side and “elbow down” on the other, so the checkboxes must be wired to the posture, not the slot.</div>
<p class="muted small">“Today” on this page = the committed viewer (<code>8bac513</code>). The viewer agent's uncommitted work, seen in its working tree while this was built, already adds Show = All, colour by consistency, Show following Colour by, Escape to deselect and an angular-resolution line; these proposals sit on top of that and do not depend on it.</p>
<div class="verdict"><b>New · <a href="#g2">G′ — five more API proposals (round 2)</a></b>: attribute namespace, query builder, numpy masks, sampler-first and a table of IK solutions, each run on the UR5e. <b>Pick: P1</b> — <code>ws.dexterous.sample(3, rng)</code>, <code>ws.hemisphere.inside(belt).only(elbow="up").sample_pairs(3, rng)</code> — with P2's query as its saved form.</div>
<div class="selfcheck"></div>
<table><tr><th></th><th>Ask</th><th>Recommended</th><th>Why</th></tr>{pick_rows}</table>

<h2 id="h">H · What is the FK cloud, and what do its colours mean?</h2>
<div class="answer"><div>
<p><b>Yes — the FK cloud is what sized the voxel grid.</b> Verified in <code>analyze.py</code> / <code>sampling.py</code> / <code>arm.py</code> and the saved arrays:</p>
<ol>
<li><b>Sample joints:</b> {fmt(f['n_fk'])} joint vectors, uniform inside the joint limits (seed {cfg['seed']}). FK gives each one a tool position.</li>
<li><b>Make the box:</b> the bounding box of <i>all</i> those positions (colliding ones too) is the grid box: {box['size'][0]:.2f} × {box['size'][1]:.2f} × {box['size'][2]:.2f} m.
The step is the extent ÷ 18 = {f['step_mm']:.0f} mm, so 19 × 19 × 19 = {fmt(f['P'])} grid points.</li>
<li><b>Ask IK there:</b> every grid point × the {f['O']} shared orientations × every IK branch. That is the voxel map; the cloud is only drawn for context.</li>
</ol>
<p><b>Colours</b> are each sample's joint-space check (<code>Arm.check</code>, the first failure wins: joint limit → FK fails → collision → torque → singular → OK):</p>
<ul>
<li><span class="sw" style="background:rgb(46,160,67)"></span> <b>OK</b> {fmt(cc.get('OK', 0))}</li>
<li><span class="sw" style="background:rgb(220,53,69)"></span> <b>collision</b> {fmt(cc.get('COLLISION', 0))} — for the UR5e the only collision model is the floor: some skeleton point is below z = 0. That is why the lower half of the cloud is red.</li>
<li><span class="sw" style="background:rgb(124,77,255)"></span> <b>singular</b> {fmt(cc.get('SINGULAR', 0))} — manipulability under 1 % of the arm's typical (95th-percentile) value.</li>
<li class="muted">Orange (over torque) and grey (cannot close) never occur on the UR5e: it has no torque limits or closed loop in this model.</li>
</ul>
<p class="muted small">The viewer draws 12,000 of the 20,000, evenly thinned. Gap: the Workspace legend explains the voxel colours only, so nothing on screen explains the cloud's. Fix: when <b>FK cloud</b> is ticked, add a second legend line “cloud: ● OK ● collision (floor) ● singular”.</p>
</div><div><canvas id="h-cloud" style="height:330px"></canvas><div class="muted small">Every 3rd FK sample (6,000), its code colour, and the dashed grid box it produced.</div></div></div>

<h2 id="a">A · Choosing which IK solution types to include</h2>
<div class="ask">You: “fix a solution type … select whether I want to support elbow up or down, wrist up or down, the other thing … all 8 or any combination … a standard Robot MSG that allows you to name the different IK solutions … so you can automatically generate these check box UI.”</div>
<p><b>The type</b> (prototype: <code>docs/proposals/ik_branches.py</code>). One small frozen message names the questions; the arm tags every solution with its answers. All three UIs below are generated from <code>UR5E_BRANCHES.ui_spec()</code> alone; nothing in them knows it is a UR.</p>
<pre>@dataclass(frozen=True)
class BranchAxis:
    name: str                    # "elbow"
    options: tuple[str, ...]     # ("up", "down"): option i is tag value i
    joint: str = ""              # the joint that flips it, shown as "elbow (J3)"
    rule: str = ""               # how a solution is tagged, in plain words (tooltip)

@dataclass(frozen=True)
class IKBranches:                # would live in robot_msgs.manipulation, beside NamedPose
    arm: str
    joint_names: tuple[str, ...]
    axes: tuple[BranchAxis, ...]
    def type_of(self, tags) -> Array        # (N, n_axes) answers -> (N,) type index
    def type_name(self, i) -> str           # "shoulder front · elbow up · wrist J5+"
    def allowed(self, **choices) -> Array   # allowed(elbow="up") -> (n_types,) bool
    def ui_spec(self) -> dict               # what a viewer draws: axes, options, types

UR5E_BRANCHES = IKBranches("ur5e", JOINTS, (
    BranchAxis("shoulder", ("front", "back"), "shoulder_pan", "front: the wrist is in front of the pan axis"),
    BranchAxis("elbow", ("up", "down"), "elbow", "up: the elbow is above the shoulder-to-wrist line"),
    BranchAxis("wrist", ("J5+", "J5−"), "wrist_2", "the sign of wrist_2 (J5): the wrist flip")))

# beside ReachMap.sol_q: one tag per solution, computed from its joints by the arm
ReachMap.sol_type: Array   # (K,) uint8, IKBranches type index
Arm.branches: IKBranches | None ;  Arm.tag(q) -> (N, n_axes)</pre>
<p><b>Why tags come from the posture, not the solver's slot</b> (measured on 20,000 random UR5e solutions): with the shoulder on one side, slot “elbow +” is physically elbow-up 99.5 % of the time; on the other side it is elbow-up 0.7 % of the time. A checkbox wired to the slot would mean the opposite half the time, and a numeric IK (the URDF path) has no stable slots at all.
Solutions per type: {', '.join(fmt(x) for x in f['per_type'])}.</p>
<p class="muted small">What the filter does on this arm (measured): elbow-up only keeps every level ({lv_line(c['elbow_up'])}); front + elbow up keeps {lv_line(c['front_up'])}; one single type keeps {lv_line(c['one_type'])}. With only a floor check, the UR's elbow pair is interchangeable; the filter matters once a real collision model or a task (“never flip the wrist”) is in.</p>
<div class="grid">
{card("A1 One row per question", "Three rows (shoulder J1, elbow J3, wrist J5), two toggles each; allowed = every combination of the ticked answers.",
      "<div id='a1'><div class='vpanel'><div class='vfold'>Allowed IK types</div></div><canvas></canvas><div class='readout'></div></div>",
      "you think in questions (“never elbow down”)", "you want two specific types only (front-up-J5+ and back-down-J5−): 27 of the 255 combinations are reachable")}
{card("A2 Eight posture tiles", "A 2 × 4 grid: rows = elbow, columns = shoulder × wrist; each tile is the arm in that posture at the selected point; click a tile, a row or a column.",
      "<div id='a2'><div class='vpanel'><div class='vfold'>Allowed IK types</div><div class='tiles'></div></div><canvas class='cloud'></canvas><div class='readout'></div></div>",
      "you want to SEE what each type looks like, and any combination", "an arm has more than ~16 types (each tile needs a posture to draw)", picked=True)}
{card("A3 Filter by example", f"Under Selected point, every solution carries its answers as chips; click a chip to keep only that kind. Selected point ({sel[0]:+.3f}, {sel[1]:+.3f}, {sel[2]:+.3f}) m: the grid point nearest the one in your screenshot, with the same 924 working pairs.",
      "<div id='a3'><div class='vpanel'><div class='vfold'>Selected point</div><div class='req'></div><div class='sols'></div></div><canvas></canvas><div class='readout'></div></div>",
      "you discover the types while clicking through solutions", "nothing is selected: the filter is hidden until you click a voxel")}
</div>
{prune([("Any combination", 30, "you: 'all 8 or any combination thereof'"), ("Obvious at a glance", 30, "what 'elbow up' means is visible, not looked up"),
        ("Generated for any arm", 25, "the same UI from IKBranches for RR (2 types), shoulder3 (4), five-bar (4), UR (8)"), ("Panel space", 15, "fits the 360 px side panel")],
       [("A1 One row per question", [2, 3, 5, 5]), ("A2 Eight posture tiles", [5, 5, 4, 3]), ("A3 Filter by example", [2, 3, 4, 4])], "A2 Eight posture tiles")}
<p><b>Pick A2</b>, with A1's speed kept: the row and column headers toggle a whole question at once. Splice A3's chips into the Selected-point line so each solution says what type it is.</p>

<h2 id="b">B · Joint ranges: limits vs what the working solutions use</h2>
<div class="ask">You: “histogram or violin plot could be clearer than using the colour, as height difference is easier to perceive than colour.” Also: always write (J1/J2/…) beside the joint name.</div>
<p class="muted small">All five from the same {fmt(f['n_sol'])} working UR5e solutions. <b>Finding:</b> every joint's solutions sit inside about ±180° ({rf_text}) although the limits are ±360°: the IK returns angles wrapped to ±180° and the library only uses the ±360° twin when a value is outside the limits. So the “used range” shows the solver's habit, not a need — the plot should say so, or draw ±360° joints on a −180…180° axis. Likewise wrist_3's spikes every 60° are the six sampled rolls, not a preference of the arm.</p>
<div class="grid two">
{card("B1 Violin", "Width = how many solutions use that angle; one violin per joint over its limit bar.", f"<img class='plot' src='{B['violin']}'>",
      "you want shape and limits in one compact row per joint", "two thin peaks: a violin smooths them together")}
{card("B2 One histogram per joint", "Bar height = share of solutions in each 10° bin; the joint limit is the grey band behind.", f"<img class='plot' src='{B['hist']}'>",
      "you want counts you can read by height (your point), gaps included", "the panel is short: six rows need ~260 px", picked=True)}
{card("B3 Ridgeline", "The same histograms overlapped as ridges: a denser stack.", f"<img class='plot' src='{B['ridge']}'>",
      "you compare the shape of many joints", "ridges overlap and hide each other's tops")}
{card("B4 Range bars", "Box-plot style: 1–99 % band, middle half, median, and ‘uses N % of the limit’.", f"<img class='plot' src='{B['box']}'>",
      "you only need ‘how much of the limit is used’", "the solutions have a gap: a bar paints straight over it")}
{card("B5 Split by IK type", "Up = elbow-up solutions, down (mirrored) = elbow-down: ties B to the A filter.", f"<img class='plot' src='{B['split']}'>",
      "you are choosing IK types and want to see what each costs in range", "you just want the overall range: two halves to read")}
</div>
{prune([("Height, not colour", 30, "your reason for the change"), ("Limits vs used, visible", 25, "the question the plot answers"),
        ("Shape and gaps show", 25, "a joint whose solutions avoid a band must show it"), ("Compact", 20, "fits under Joint space")],
       [("B1 Violin", [4, 4, 4, 4]), ("B2 One histogram per joint", [5, 5, 5, 3]), ("B3 Ridgeline", [5, 3, 5, 3]), ("B4 Range bars", [3, 5, 1, 5]), ("B5 Split by IK type", [5, 4, 5, 2])],
       "B2 One histogram per joint")}
<p><b>Pick B2</b>; switch it to B5 automatically while an IK-type filter (A) is active, so the plot shows what the filter costs.</p>

<h2 id="c">C · An Ask-here box you can tilt, with the angles spelled out</h2>
<div class="ask">You: “make the angle range and step size more explicit inside the box … you should also be able to tilt it in the different axes, and we use the Z axis of the box as the Z axis of the workspace.”</div>
<p class="muted small">Spikes = the approach directions asked at the box centre (the real ring algorithm of <code>OrientationSet.sample</code>, ported to the page and checked below), approach = −box Z; blue = straight along box Z (the 4-DOF ring). In the viewer they turn green/grey by IK, as Drop POI does today. Today the gizmo is created with <code>disable_rotations=True</code> and the approach is a fixed dropdown.</p>
<div class="grid">
{card("C1 Drag to tilt, numbers mirror it", "Turn on the gizmo's rotation rings. The panel reads back the tilt and lists the angle set; editing a number moves the gizmo too.",
      "<div id='c1'><div class='vpanel'><div class='vfold'>Ask here</div>"
      "<div class='vrow'><label>Tilt about X <span class='muted small'>(gizmo ring)</span></label><div><input type='range' data-k='rx' min='-90' max='90' value='20'><span data-k='rxv' class='small'></span></div></div>"
      "<div class='vrow'><label>Tilt about Y <span class='muted small'>(gizmo ring)</span></label><div><input type='range' data-k='ry' min='-90' max='90' value='0'><span data-k='ryv' class='small'></span></div></div>"
      "<div class='vrow'><span class='lab'>Box Z</span><span class='tiltread small'></span></div>"
      "<div class='vrow'><label>Angles up to (from box Z)</label><select data-k='max'><option value='0'>0° straight only</option><option value='45'>45°</option><option value='90' selected>90° hemisphere</option><option value='180'>180° everything</option></select></div>"
      "<div class='vrow'><label>Tilt step (°)</label><input type='number' data-k='step' value='45' min='5' max='90' step='5'></div>"
      "<div class='vrow'><label>Roll step (°)</label><input type='number' data-k='roll' value='60' min='10' max='180' step='10'></div></div>"
      "<canvas></canvas><div class='readout'></div></div>",
      "you set the tilt by eye against the scene (a tilted belt)", "you need an exact, repeatable angle: type it in the mirrored field", picked=True)}
{card("C2 Type the box frame", "Centre, roll / pitch / yaw and the angle set as numbers; a copyable Python line shows exactly what will run.",
      "<div id='c2'><div class='vpanel'><div class='vfold'>Ask here</div>"
      "<div class='vrow'><label>Roll / pitch / yaw (°)</label><div class='trio'><input type='number' data-k='roll' value='0' step='5'><input type='number' data-k='pitch' value='25' step='5'><input type='number' data-k='yaw' value='0' step='5'></div></div>"
      "<div class='vrow'><label>Angles up to (°)</label><input type='number' data-k='max' value='90' min='0' max='180' step='15'></div>"
      "<div class='vrow'><label>Tilt step (°)</label><input type='number' data-k='step' value='30' min='5' max='90' step='5'></div>"
      "<div class='vrow'><label>Roll step (°)</label><input type='number' data-k='rollstep' value='60' min='10' max='180' step='10'></div>"
      "<pre class='py' style='white-space:pre-wrap;font-size:11px'></pre></div><canvas></canvas><div class='readout'></div></div>",
      "the tilt comes from a drawing or a measured belt angle", "you just want to try a few tilts quickly")}
{card("C3 Presets", "Three chip rows: which way box Z points, how wide the cone of approaches is, how fine. No numbers unless you ask.",
      "<div id='c3'><div class='vpanel'><div class='vfold'>Ask here</div>"
      "<div class='vrow'><span class='lab'>Box Z points</span><div class='pills'><button class='preset on' data-group='up' data-v='[0,0]'>up</button><button class='preset' data-group='up' data-v='[20,0]'>tilted 20°</button><button class='preset' data-group='up' data-v='[0,45]'>tilted 45°</button></div></div>"
      "<div class='vrow'><span class='lab'>Approaches</span><div class='pills'><button class='preset' data-group='cone' data-v='0'>straight</button><button class='preset' data-group='cone' data-v='45'>±45°</button><button class='preset on' data-group='cone' data-v='90'>hemisphere</button><button class='preset' data-group='cone' data-v='180'>all</button></div></div>"
      "<div class='vrow'><span class='lab'>Fineness</span><div class='pills'><button class='preset on' data-group='fine' data-v='[45,60]'>coarse 45°/60°</button><button class='preset' data-group='fine' data-v='[30,45]'>30°/45°</button><button class='preset' data-group='fine' data-v='[15,30]'>fine 15°/30°</button></div></div>"
      "</div><canvas></canvas><div class='readout'></div></div>",
      "the same few setups repeat (belt flat, belt tilted)", "the tilt you need is not a preset")}
</div>
<table><tr><th>tilt step</th><th>angles up to</th><th>rolls</th><th>library count</th><th>page count</th><th></th></tr><tbody id="c-check"></tbody></table>
{prune([("Tilt in any axis, fast", 30, "your ask"), ("Angles explicit in the box", 30, "range + step visible where you look"), ("Repeatable numbers", 20, "the same box can be asked again"), ("Little to learn", 20, "no new concepts")],
       [("C1 Drag to tilt, numbers mirror it", [5, 5, 3, 4]), ("C2 Type the box frame", [3, 5, 5, 3]), ("C3 Presets", [2, 4, 4, 5])], "C1 Drag to tilt, numbers mirror it")}
<p><b>Pick C1</b>, with C2's typed fields as the mirror (drag or type, both move the gizmo). Approach = −box Z replaces the Approach dropdown; the box result should record the box pose and the angle set (see F's metadata).</p>

<h2 id="d">D · Colour schemes for the workspace points</h2>
<div class="ask">You: “5 different proposals … prior art from how past reachability things did it … one would be the standard heat map … you shouldn't need to relearn anything and it's obvious at a glance.”</div>
<div class="controls"><label>Colour by <select id="d-by"><option>pose DOF</option><option>fraction of orientations</option></select></label>
<label>See it as <select id="d-sim"><option value="none">normal vision</option><option value="deuteranopia">deuteranopia (green-weak)</option><option value="protanopia">protanopia (red-weak)</option><option value="tritanopia">tritanopia (blue-weak)</option></select></label>
<span class="muted small">simulation: Machado, Oliveira &amp; Fernandes 2009</span></div>
<p class="small"><b>Seen on the real cloud:</b> in 3D the lowest level wraps around everything else (the 559 “reachable only” points form the outer shell of the dome), so whatever colour means “least” covers most of the picture. That is why the pick runs light → dark: the shell stays pale and the dexterous core reads as the solid part, which is also your Nov 19 rule, “white means empty”.</p>
<div class="grid">
{''.join(f"<div class='card{' picked' if d_names[k] == d_best else ''} dview' data-scheme='{k}'><h3>{d_names[k]}{' <span class=badge>pick</span>' if d_names[k] == d_best else ''}</h3><div class='muted small'>{th}</div><canvas></canvas><div class='legend'></div><div class='trade'><b>Best when</b><span>{b}</span><b>Loses when</b><span>{lo}</span></div></div>"
         for k, th, b, lo in [
             ("traffic", "Red → amber → green, as bam_reach v1 coloured reach scores (colorize.py) and as most tools mean ‘bad → good’.", "people read red = bad, green = good instantly", "red-green colour blindness (deuteranopia ΔE 7: levels 3 and 6 look alike), and red already means collision here"),
             ("viridis", "Viridis run light-to-dark: pale yellow (least) → green → dark violet (most). matplotlib's and compas_fab's colours, turned so white = empty, solid = most.", "you need it to work for everyone, in print and on screen", "someone used to compas_fab expects yellow = most (it runs the other way there)"),
             ("rainbow", "Red (least) → yellow → cyan → blue (most): the heat map of Reuleaux and the DLR capability maps.", "you want the look of the published reachability maps", "lightness jumps around (yellow brightest), so order needs the legend; red clashes with collision"),
             ("thermal", "Pale yellow → orange → dark purple (inferno, reversed): darker = more, like a heat map on white.", "‘darker = more’ on a white background", "its orange-red middle sits near the collision red"),
             ("okabe", "Four named colours from the Okabe-Ito colour-blind palette: one per workspace name, not an amount.", "you think of the four kinds as names (as the Show dropdown does)", "order is not visible: which is ‘more’, orange or blue?"),
             ("today", "Today: white → teal → deep blue (ColorBrewer YlGnBu family); kept here as the reference.", "it already exists", "the lightest level (3) is faint on white, and its colour-blind margin is the lowest of the ordered ones")])}
</div>
<table><tr><th>Scheme</th><th class="num">ΔE normal</th><th class="num">deutan</th><th class="num">protan</th><th class="num">tritan</th><th>one direction</th><th class="num">ΔE to problem colours</th><th class="num">ΔE to white</th><th>levels 3 → 6</th></tr>{mt_rows}</table>
<p class="muted small">ΔE = smallest CIE76 colour difference between the four level colours (≥ ~20 reads as clearly different; under 10 is easy to confuse). Computed in <code>build_proposals.py</code>.</p>
{prune(d_crit, d_variants, d_best)}
<p><b>Pick {d_best}.</b> It wins on the measurements, it is the colour set compas_fab and matplotlib already use, and it leaves red / purple / orange free to mean collision / singular / torque, as the code vocabulary says. Keep the empty level near-white (your Nov 19 note: “white means empty”). For continuous colour-bys (fraction, IK solutions) use the same viridis ramp so one key reads everywhere.</p>
<details><summary><b>Prior art</b> — what each reachability tool uses (sources checked)</summary>
<table><tr><th>Tool / paper</th><th>Scale</th><th>Source</th></tr>
<tr><td>Zacharias, Borst, Hirzinger (DLR) capability maps</td><td>reachability index: red = minimum, blue = maximum</td><td><a href="https://www.csc.kth.se/~danik/RSS/rssAbstract.pdf">RSS abstract</a>; Porges thesis 2014 (HSV, 0 red → 1 blue): <a href="https://elib.dlr.de/93767/2/OP-thesis.pdf">elib.dlr.de</a></td></tr>
<tr><td>Reuleaux (ROS)</td><td>5 bands on reachability index: &lt;5 red, 5–30 yellow, 30–50 green, 50–90 cyan, ≥90 blue</td><td><a href="https://github.com/ros-industrial-attic/reuleaux/blob/kinetic-devel/workspace_visualization/src/reachability_map_visual.cpp">reachability_map_visual.cpp</a> (setColorSpherebyRI)</td></tr>
<tr><td>ROS-Industrial REACH</td><td>HSV hue 270° (blue/violet) at the lowest score → 0° (red) at the highest; unreached drawn black — the opposite direction</td><td><a href="https://github.com/ros-industrial/reach/blob/master/src/types.cpp">types.cpp</a> (computeHeatMapColors)</td></tr>
<tr><td>Vahrenkamp &amp; Asfour 2014 (Simox)</td><td>manipulability: red = high, blue = low</td><td><a href="https://h2t.iar.kit.edu/pdf/Vahrenkamp2014.pdf">Vahrenkamp2014.pdf</a></td></tr>
<tr><td>compas_fab ReachabilityMap</td><td>matplotlib viridis (score = number of IK solutions)</td><td><a href="https://gramaziokohler.github.io/compas_fab/latest/examples/07_reachability_map/01_reachability_map.html">compas_fab docs</a></td></tr>
<tr><td>bam_reach v1 (yours, Jul 2026)</td><td>red → yellow → green on reach score, 7 bins</td><td><code>bam_reach/colorize.py</code></td></tr>
<tr><td>Colour-map research</td><td>cividis for colour-blind viewers (Nuñez et al. 2018); rainbow maps hide detail (Borland &amp; Taylor 2007); Okabe-Ito palette</td><td><a href="https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0199239">PLOS ONE</a> · IEEE CG&amp;A 27(2) · <a href="https://jfly.uni-koeln.de/color/">jfly</a></td></tr>
</table><p class="muted small">Not verified: a MoveIt reachability display (none found) and RoboDK / Emulate3D colours (no primary source). The direction is not standard: DLR and Reuleaux put blue = most, REACH and Simox put red = most — one more reason to pick a scheme whose order reads from lightness.</p></details>

<h2 id="e">E · A histogram under Workspace that is the legend and the Show control</h2>
<div class="ask">You: “here it says 1462 points… show me the histogram of the breakdown … mimics as a legend … Show should also update to match what you are colouring by … be able to show or not show any of them … a range selector under the histogram.” Plus: colour by consistency (FK/IK matched), and an “All” that shows every sampled voxel.</div>
<p class="muted small">Each variant replaces the Show dropdown: its bins ARE the Colour-by bins, and “unreachable / 0” is a bin you can switch on (= your “All”: all {fmt(f['P'])} sampled points). Colours: {d_best}. <b>Consistency:</b> every saved result has 0 FK(IK) mismatches ({', '.join(f'{k} {v}' for k, v in f['saved_mismatch'].items())}), so that colour-by would be one flat colour today; the demo uses the same UR5e map recomputed with an IK whose shoulder_lift zero is {f['eps_deg']:.3f}° off the FK — the June 2025 “inconsistent IK frames” kind of bug — which gives {fmt(cons[1])} all-matched, {fmt(cons[2])} partly and {fmt(cons[3])} all-mismatched points (far points fail first: the error grows with reach).</p>
<div class="grid">{e_html}</div>
{prune([("Legend + histogram in one", 25, "you: 'mimics as a legend'"), ("Any bin on / off", 25, "you: 'show or not show any of them'"), ("Obvious at a glance", 25, "counts and colours read without hovering"),
        ("Panel space", 15, "fits under Workspace"), ("10+ bins", 10, "fraction of orientations has 11")],
       [("E1 Clickable bar legend", [5, 5, 5, 4, 4]), ("E2 Stacked strip", [3, 5, 3, 5, 3]), ("E3 Histogram + range slider", [5, 2, 4, 4, 5]), ("E4 Legend list", [3, 5, 5, 2, 3]), ("E5 “At least” threshold", [4, 2, 4, 4, 4])],
       "E1 Clickable bar legend")}
<p><b>Pick E1.</b> Add E3's two-handle range under it only for the amount-type colour-bys (fraction, IK solutions), where “between” is the natural question. The <b>Show</b> dropdown goes away; the four workspace names become the bins of Colour by = pose DOF.</p>

<h2 id="f">F · The wizard, inside the Workspace panel</h2>
<div class="ask">You: “1 select/upload the robot, 2 it auto-places the voxels by sampling the FK, 3 select resolution: voxel size (XYZ), orientation step (deg) and rz (deg), 4 runs … if you edit the values the Compute Workspace button lights up … hide the old results, but you can reset … the results should 100 % carry the information that was used to compute them.”</div>
<p class="muted small">Try it: change any number. The cost line uses the measured speed of this result ({fmt(f['P'] * f['O'])} targets in {f['seconds']:.1f} s) and the real grid rule (each side grows to whole steps around the fixed centre). The full-workspace analysis offers no box or angle range — only resolution — as you asked.</p>
<div class="grid">
{card("F1 Numbered steps", "The four steps as a short numbered list; 1 and 2 are done automatically and say what they did.",
      f"<div class='fvar' data-kind='steps'>{f1}<canvas></canvas></div>", "someone new opens it, or a URDF was just uploaded", "you only change one number: the steps take a third of the panel")}
{card("F2 Used-settings card", "The fields show what the result was computed with. Edit one: it turns amber, ‘was 45’ appears, Compute workspace lights up, the old map hides; Reset goes back.",
      f"<div class='fvar' data-kind='diff'>{f2}<canvas></canvas></div>", "you iterate on resolution (your described flow)", "a brand-new user might not know steps 1–2 happened (one line in the header says it)", picked=True)}
{card("F3 Recipe line + drawer", "One line says how the result was made; Edit opens the fields underneath.",
      f"<div class='fvar' data-kind='recipe'>{f3}<canvas></canvas></div>", "the panel is crowded and the settings rarely change", "the fields are hidden, so ‘what did this use?’ costs a click")}
</div>
{prune([("Shows what the result used", 30, "you: 'either showing you the values it used OR you can adjust them'"), ("Edit → light up, hide old, reset", 30, "your described behaviour"),
        ("Teaches the 4 steps", 20, "the reason wizards are nice"), ("Panel space", 20, "Workspace already holds Show, Colour by, cut, blob")],
       [("F1 Numbered steps", [4, 5, 5, 3]), ("F2 Used-settings card", [5, 5, 3, 5]), ("F3 Recipe line + drawer", [4, 4, 3, 5])], "F2 Used-settings card")}
<p><b>Pick F2</b>, with F1's one-line header (“robot ur5e · voxels placed from {fmt(f['n_fk'])} FK samples”) so steps 1–2 stay visible.</p>
<h3>The metadata the result must carry</h3>
<p>Today <code>result.json["config"]</code> holds: step {cfg['step']:.4f} m (one number, not XYZ), box centre/size, n_fk {cfg['n_fk']}, floor_z {cfg['floor_z']}, singular_tol {cfg['singular_tol']}, w_ref, seed {cfg['seed']}, and <b>orientations: {cfg['orientations']} — only the count</b>. The tilt step, roll count, max tilt and approach that made those 132 are not recorded, nor the round-trip tolerances (they sit in the npz's meta), the arm's version or the code commit. So a result cannot yet fill the F2 card by itself. Proposed:</p>
<pre>@dataclass(frozen=True)
class ReachSettings:
    \"\"\"What a reach result was computed with. Saved as result.json["settings"]; the viewer's
    Workspace card is filled from it, and a change of digest() is what lights Compute workspace.\"\"\"
    # 1 robot
    arm: str                                   # "ur5e"
    arm_source: str                            # "robot_reach.robots:ur5e" or the URDF path
    arm_digest: str                            # sha256[:16] of the URDF text / factory kwargs
    ik: str                                    # "closed form, 8 branches" | "numeric DLS, 8 seeds"
    # 2 where to look: automatic, recorded not chosen
    fk_samples: int = 20_000
    seed: int = 0
    box_center_m: tuple[float, float, float] = (-0.006, 0.009, 0.166)
    box_size_m: tuple[float, float, float] = (1.897, 1.900, 1.902)
    # 3 resolution: the only things a person sets
    voxel_m: tuple[float, float, float] = ({cfg['step']:.4f}, {cfg['step']:.4f}, {cfg['step']:.4f})
    tilt_step_deg: float = 45.0
    roll_step_deg: float = 60.0
    max_tilt_deg: float = 180.0                # a full analysis always asks everything
    approach: tuple[float, float, float] = (0.0, 0.0, -1.0)
    # checks
    floor_z_m: float | None = 0.0
    singular_tol: float = 0.01                 # of w_ref, the arm's 95th-percentile manipulability
    pos_tol_m: float = 1e-4
    rot_tol_deg: float = 0.057
    # provenance (not part of digest)
    library: str = "robot_reach 0.1.0 @ &lt;commit&gt;"
    computed_at: str = "<ISO 8601 time>"
    seconds: float = {f['seconds']:.1f}

    def digest(self) -> str: ...              # hash of sections 1-3 + checks: equal digest = same result</pre>
<p class="muted small">A box result (C) carries the same object with its own box pose and angle set, so every layer says how it was made.</p>

{f.get('round2_html', '')}
<h2 id="g">G · The API downstream code uses to get a workspace</h2>
<p class="small"><b>Round 1, kept for the record — superseded by <a href="#g2">G′</a></b>: round 2 renames “branch” to IK configuration and puts every workspace kind on one object.</p>
<div class="ask">You: “now that I can reliably go from robot description to any of its workspaces … go like .sample() … What is the final API and robot_msg type … 3 proposals … How does this compare to what I already had?”</div>
<div class="grid" style="grid-template-columns:1fr">
{card("G1 A Workspace object in robot_reach", "Methods hang off the result. Simple to build; every consumer imports robot_reach.",
      "<pre>import robot_reach as rr\nresult = rr.load(\"out/ur5e\")\nws = result.workspace(\"hemisphere\", allow=dict(elbow=\"up\"))\nws.sample(100, rng)        # Samples(T, q, branch_type)\nws.contains(points)        # (N,) bool\nws.region(rr.Box(...))     # a smaller Workspace\nws.sample_pairs(50, rng, same_type=True)\nws.nearest_ok_pose(T)      # (T_ok, q, m, deg)</pre>",
      "one repo uses it", "robot_trajectories would import robot_reach (a leaf importing a leaf, which your robot_msgs rule forbids)")}
{card("G2 A ReachableWorkspace value in robot_msgs", "A frozen message of plain arrays (voxels, orientations, the checked joint solutions, their IK types, the settings). robot_reach makes it; consumers need only robot_msgs + numpy.",
      "<pre>@dataclass(frozen=True, eq=False, repr=False, kw_only=True)\nclass ReachableWorkspace(Msg):           # robot_msgs.manipulation\n    arm: str; kind: str; frame: str\n    joint_names: tuple[str, ...]\n    positions: Array      # (P, 3) voxel centres, m\n    voxel_size: Array     # (3,) m\n    orientations: Array   # (O, 3, 3) the kind's orientation set\n    ok: Array             # (P, O) bool\n    solution_pose: Array  # (K,) p * O + o, sorted\n    solutions: Array      # (K, dof) rad: checked joints per pose\n    solution_type: Array  # (K,) index into branches\n    branches: IKBranches | None\n    settings: ReachSettings\n    def contains(self, points) -&gt; Array\n    def region(self, shape) -&gt; ReachableWorkspace\n    def sample(self, n, rng) -&gt; PoseSamples\n    def sample_pairs(self, n, rng, *, same_type=True)\n    def nearest_ok_pose(self, pose) -&gt; tuple[PoseSample, float, float]\n    def voxels(self) -&gt; tuple[Array, Array, Array]\n    def joint_set(self) -&gt; Array          # G3's view, for free\n\n# producer\nws = rr.load(\"out/ur5e\").workspace(\"hemisphere\", types=UR5E_BRANCHES.allowed(elbow=\"up\"))\nws.save(\"belt_arm.ws.npz\")\n# consumer (robot_trajectories): robot_msgs only\nws = load_npz(\"belt_arm.ws.npz\").region(belt)\na, b = ws.sample_pairs(100, rng)</pre>",
      "several packages consume workspaces (trajectory generation, sim2real coverage, the facility composer)", "it needs one more robot_msgs release before anyone can use it", picked=True)}
{card("G3 A joint-space set", "Hand consumers the checked joint configurations (K × dof, each with its tool pose and type); Cartesian questions go through stored poses. Extends what robot_trajectories already reads.",
      "<pre>configs = rr.load(\"out/ur5e\").joint_set(\"hemisphere\", types=...)\nconfigs.q           # (K, 6) rad\nconfigs.tool_T      # (K, 4, 4)\nconfigs.type        # (K,)\nspace = JointSpace.from_limits(limits).with_reach_set(configs)\nq_a, q_b = space.sample(2, rng)   # stays inside the reachable set</pre>",
      "the generator thinks in joints (excitation, coverage trajectories)", "you ask ‘is this belt point in the dexterous workspace?’: needs a spatial index on top")}
</div>
{prune([("Consumer never imports robot_reach", 30, "your rule: every package depends on robot_msgs; leaves never import each other"),
        ("The verbs he asked for", 25, "sample, contains, region, nearest_ok_pose, A→B pairs, IK-type aware"),
        ("Only promises what was checked", 20, "every pose returned was IK'd, limit-checked, FK round-tripped"),
        ("Fits what exists", 15, "robot_trajectories' ReachMapLike (joint_names, joint_solutions, valid_mask) keeps working"), ("Saves without the code", 10, "npz, no pickle")],
       [("G1 A Workspace object in robot_reach", [1, 5, 5, 4, 3]), ("G2 A ReachableWorkspace value in robot_msgs", [5, 5, 5, 4, 5]), ("G3 A joint-space set", [4, 2, 5, 5, 4])],
       "G2 A ReachableWorkspace value in robot_msgs")}
<p><b>Pick G2</b> with G1's verbs as its methods and G3 as one of them (<code>joint_set()</code>). The prototype (<code>docs/proposals/workspace_api.py</code>, numpy only, same shape) ran on the real result — a belt zone 0.30–0.65 m in front, ±0.40 m across, 0.05–0.40 m up; hemisphere workspace; elbow up only:</p>
<pre>{esc(f['demo'])}</pre>
<p class="muted small">Every number above is printed by <code>docs/proposals/api_demo.py</code>. The FK check re-runs the arm's FK on 2,000 sampled joint vectors: the poses <code>sample()</code> hands out are exactly reached. <code>nearest_ok_pose</code> moved a target above the belt zone to the nearest checked voxel, 102 mm away, with the same pointing-down orientation.</p>
<h3>Against what you already had</h3>
<table><tr><th></th><th>bam_reach v1 (Jul 2026)</th><th>robot_trajectories today</th><th>Sample joints, FK, go (your simple way)</th><th>G2 (proposed)</th></tr>
<tr><td>What you get</td><td><code>WorkspaceRegion.sample_pair(seed)</code> → one (position, orientation) index; <code>PoseSpace.from_workspace_region(...).sample(rng)</code> → a pose + its first joint solution</td><td><code>ReachSet.from_reach_map(map)</code> reads <code>joint_solutions</code> (P, O, J, first branch) + <code>valid_mask</code></td><td>a random q and its FK pose</td><td>poses + every checked solution + its IK type</td></tr>
<tr><td>Only inside a task zone (belt)</td><td>no (whole map)</td><td>no</td><td>by rejection: 0.07 % of draws land over the belt pointing down and clear the floor</td><td><code>region(shape)</code>: 100 % by construction</td></tr>
<tr><td>Pick the IK type</td><td>no (first branch only)</td><td>no (first branch only)</td><td>no</td><td>yes (<code>types=</code>, <code>same_type=</code>)</td></tr>
<tr><td>A → B pairs</td><td>the dog-leg campaign draws pairs with a minimum distance (campaign code, five-bar only)</td><td>no pairs; its trajectory kinds stay inside the reachable joint set and <code>workspace_sweep</code> tours it</td><td>pick any two</td><td><code>sample_pairs</code>, same type by default</td></tr>
<tr><td>contains / nearest</td><td>masks only</td><td><code>nearest_is_reachable(q)</code>, in joint space</td><td>no</td><td>yes</td></tr>
<tr><td>Needs robot_reach at runtime</td><td>yes (bam_reach)</td><td>no (duck-typed)</td><td>needs the arm's FK</td><td>no (robot_msgs)</td></tr>
</table>
<p><b>When the simple way is enough:</b> any trajectory where any reachable pose will do — excitation, coverage, “move somewhere and back” — robot_trajectories already samples joint space for those, and needs no map. <b>When it is not:</b> poses must sit in a task zone with an approach (over the belt, pointing down): only 147 of 200,000 random joint draws qualify (0.07 %; 0.74 % with the looser hemisphere rule), and two random draws are far apart in joints (largest joint move median 168° vs 120° for same-type workspace pairs) — the “strange wrist rotations” you expected. <b>Neither checks the path between A and B</b> (swept-path collision is still not built); the workspace guarantees the end poses only.</p>

<h2>What was not done</h2>
<ul class="small">
<li>No change to <code>src/</code> or the running viewer (another agent owns it). The mock-ups are this page's HTML on the real data, not viser; <b>NOT VERIFIED IN-APP</b>.</li>
<li>The consistency colour uses a deliberately offset IK, because every saved result has zero mismatches.</li>
<li>C's spikes show the directions asked, not reachability (that needs the arm's IK at the box pose, which the page does not run).</li>
<li>F's Compute button is a mock (it records the new settings; it does not run <code>analyze()</code>). The cost estimate scales the measured speed of this result linearly.</li>
<li>RR / five-bar versions of A were not drawn; the type handles them (2 and 4 types) but their tag rules were not written.</li>
</ul>
</main>
<script>window.DATA_B64="{blob}";window.MASK_B64="{mask_blob}";window.EPS_DEG="{f['eps_deg']:.3f}";</script>
<script>{js}</script></body></html>"""


def f_schemes(key):  # filled by render's caller through module state (set below)
    return _SCHEMES[key]["levels"][1:]


_SCHEMES: dict = {}

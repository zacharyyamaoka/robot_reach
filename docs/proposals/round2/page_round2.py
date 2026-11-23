"""HTML for section G′ (round 2 of the downstream API). `render(r2)` gets run_round2.run()'s dict, so
every snippet and output on the page is the exact string that was executed and what it printed."""

from __future__ import annotations

import html

PRUNE_CRITERIA = [  # fixed from the ask before the prototypes were scored
    ("Reads like your words", 30, "ws.dexterous.sample(), the full and the reachable workspace, A→B pairs, waypoints: how close the happy paths are to the sentence"),
    ("IK-configuration filter can't overclaim", 20, "elbow-up-only must re-judge which voxels are hemisphere, not drop rows from a hemisphere judged with every configuration"),
    ("Multi-chain without breaking one chain", 15, "you: “we can keep 1 robot arm, but … we do want to support multi at some point”"),
    ("Light core, robot_msgs at the seam", 15, "you: “if inside is pure numpy thats ideal! and we only need robot_msgs for like the interfaces”; consumer never imports robot_reach"),
    ("Answers more than sampling", 10, "full vs reachable, contains, counts per configuration, a custom approach cone"),
    ("Scales to finer grids", 10, "bytes grow with checked solutions (sparse), not with P × O × C (dense) or K × strings"),
]
SCORES = {"P1": [5, 5, 4, 5, 3, 5], "P2": [3, 5, 5, 5, 5, 5], "P3": [2, 4, 3, 4, 5, 2], "P4": [4, 4, 4, 5, 1, 4], "P5": [3, 2, 4, 3, 4, 2]}
ONE_LINERS = {  # the core expression of each job, as each shape spells it (copied from the snippets that ran)
    "P1": ["ws.dexterous.sample(3, rng)", "ws.hemisphere.inside(belt).only(elbow=\"up\").sample_pairs(3, rng)",
           "ws.top_down.inside(belt).only(wrist=\"J5+\").sample_waypoints(3, 5, rng)", "(cell.left.hemisphere & cell.right.hemisphere).sample(2, rng)"],
    "P2": ["ws.sample(Q.dexterous, 3, rng)", "ws.sample(Q.hemisphere.inside(belt).configuration(elbow=\"up\"), 3, rng, k=2)",
           "ws.sample(Q.top_down.inside(belt).configuration(wrist=\"J5+\"), 3, rng, k=5)", "cell.sample_shared([Q.hemisphere.on(\"left\"), Q.hemisphere.on(\"right\")], 2, rng)"],
    "P3": ["ws.sample(ws.poses(ws.dexterous, \"dexterous\"), 3, rng)", "up = ws.ok & ws.configuration(elbow=\"up\")<br>ws.sample(ws.poses(ws.voxels(\"hemisphere\", up) & ws.inside(belt), \"hemisphere\", up), 3, rng, k=2)",
           "j5 = ws.ok & ws.configuration(wrist=\"J5+\")<br>ws.sample(ws.poses(ws.voxels(\"top_down\", j5) & ws.inside(belt), \"top_down\", j5), 3, rng, k=5)", "cell.sample_shared(cell.voxels(\"hemisphere\").all(0), \"hemisphere\", 2, rng)"],
    "P4": ["sampler(reach, \"dexterous\").poses(3, rng)", "sampler(reach, \"hemisphere\", inside=belt, configurations=dict(elbow=\"up\")).pairs(3, rng)",
           "sampler(reach, \"top_down\", inside=belt, configurations=dict(wrist=\"J5+\")).waypoints(3, 5, rng)", "CoupledSampler({\"left\": sampler(left, \"hemisphere\"), …}).draw(2, rng)"],
    "P5": ["t[t.in_dexterous].sample(3, rng)", "b = t[(t.elbow == \"up\") & belt.contains(t.xyz) & t.o_hemisphere]<br>b[b.complete(\"hemisphere\")].sample_groups(\"configuration\", n=3, k=2, rng=rng)",
           "c = t[(t.wrist == \"J5+\") & belt.contains(t.xyz) & t.o_top_down]<br>c[c.complete(\"top_down\")].sample_groups(\"configuration\", n=3, k=5, rng=rng)", "c = SolutionTable.concat([…]); h = c[c.in_hemisphere & c.o_hemisphere]; h[h.shared_by(\"chain\")]"],
}


def esc(s) -> str:
    return html.escape(str(s))


def _fk(fk: dict) -> str:
    if not fk.get("n"):
        return "no poses drawn (counts only)"
    return f"FK check: {fk['n']} drawn joint vectors re-run through the arm's FK → worst {fk['pos_mm']:.1e} mm, {fk['rot_deg']:.1e}°"


def _weighted(scores) -> float:
    return sum(s * w for s, (_, w, _) in zip(scores, PRUNE_CRITERIA)) / 100


def render(r2: dict) -> str:
    f = r2["facts"]
    by = f["bytes"]
    props = r2["proposals"]
    best = max(props, key=lambda p: _weighted(SCORES[p["id"]]))["id"]
    rml = r2["reach_map_like"]

    cards = []
    for p in props:
        runs = []
        for i, (run, (title, why)) in enumerate(zip(p["runs"], r2["tasks"])):
            runs.append(f"<div class='r2run'><div class='small'><b>{i + 1}. {esc(title)}</b> <span class='muted'>— {esc(why)}</span></div>"
                        f"<pre class='code'>{esc(run['code'])}</pre><pre class='out'>{esc(run['output'])}</pre>"
                        f"<div class='muted small'>{_fk(run['fk'])} · {run['seconds'] * 1000:.0f} ms</div></div>")
        picked = p["id"] == best
        cards.append(
            f"<div class='card{' picked' if picked else ''}' id='g2-{p['id'].lower()}'><h3>{p['id']} · {esc(p['name'])}{' <span class=badge>pick</span>' if picked else ''}</h3>"
            f"<div class='muted small'>{esc(p['thesis'])}</div>"
            f"<details open><summary class='small'><b>robot_msgs dataclasses and signatures</b> (prototype: <code>docs/proposals/round2/{p['module']}.py</code>)</summary><pre>{esc(p['msgs'])}</pre></details>"
            f"{''.join(runs)}"
            f"<div class='trade'><b>IK configurations</b><span>{p['config']}</span><b>Several chains</b><span>{p['multi']}</span>"
            f"<b>Honest weakness</b><span>{p['weak']}</span></div></div>")

    head = "".join(f"<th>{esc(t)}</th>" for t, _ in r2["tasks"])
    side = "".join(f"<tr{' class=pick' if p['id'] == best else ''}><td><a href='#g2-{p['id'].lower()}'>{p['id']} {esc(p['name'])}</a></td>"
                   + "".join(f"<td><code>{x}</code></td>" for x in ONE_LINERS[p["id"]]) + "</tr>" for p in props)

    crit_head = "".join(f"<th class='num' title='{esc(why)}'>{esc(n)}<br><span class='muted'>{w} %</span></th>" for n, w, why in PRUNE_CRITERIA)
    crit_rows = "".join(
        f"<tr{' class=pick' if p['id'] == best else ''}><td>{p['id']} {esc(p['name'])}{' <span class=badge>pick</span>' if p['id'] == best else ''}</td>"
        + "".join(f"<td class='num'>{s}</td>" for s in SCORES[p["id"]]) + f"<td class='num'><b>{_weighted(SCORES[p['id']]):.2f}</b></td></tr>" for p in props)
    crit_why = "".join(f"<li><b>{esc(n)}</b> ({w} %): {esc(y)}</li>" for n, w, y in PRUNE_CRITERIA)

    worst = max(r["fk"].get("pos_mm", 0) for p in props for r in p["runs"])
    n_fk = sum(r["fk"].get("n", 0) for p in props for r in p["runs"])
    p1 = next(p for p in props if p["id"] == "P1")
    extras = "".join(f"<div class='r2run'><pre class='code'>{esc(e['code'])}</pre><pre class='out'>{esc(e['output'])}</pre>"
                     f"<div class='muted small'>{_fk(e['fk'])}</div></div>" for e in p1.get("extras", []))
    mb = lambda k: f"{by[k] / 1e6:.0f} MB"  # noqa: E731
    per_cfg = ", ".join(f"{lb} {n:,}" for lb, n in zip(f["labels"], f["per_configuration"]))

    return f"""
<style>.r2run{{margin:.6em 0}} .r2run pre.code{{margin:.2em 0 0;border-bottom-left-radius:0;border-bottom-right-radius:0}}
.r2run pre.out{{margin:0;background:#f6f8fa;color:#24292f;border-top:1px dashed #d0d7de;border-top-left-radius:0;border-top-right-radius:0;font-size:12px}}
.scroll{{overflow-x:auto}} .scroll code{{word-break:break-word}} #g2 ~ .verdict code{{word-break:break-word}}</style>
<h2 id="g2">G′ · Five more API proposals (round 2)</h2>
<div class="ask">You: “I am not completely in love with your API … the vocabulary is like around Ik Solutions … Ik Configuration … reachableworkspace.dexterous/.hemisphere.sample() … if inside is pure numpy thats ideal! and we only need robot_msgs for like the interfaces.”</div>
<div class="verdict"><b>Pick P1, the attribute namespace — <code>ws.dexterous.sample(3, rng)</code> is the actual call.</b> Fold P2's query into it as the region's data, so a region can be saved as JSON and can take a custom approach cone. Hand P4's sampler out as an optional file export.
All five ran on the real UR5e result (<code>out/ur5e</code>, read-only). P1, P2 and P3 drew the same poses from the same seed, and the FK check passed on every pose any snippet drew (worst {worst:.0e} mm, {n_fk:,} joint vectors).
P1 is also the shape you already had: bam_reach v1's <code>WorkspaceMap</code> exposed <code>.reachable / .four_dof / .hemisphere / .dexterous</code> as properties; round 1's G2 dropped that.</div>

<h3>Vocabulary used throughout</h3>
<ul class="small">
<li><b>IK solution</b>: one joint vector that reaches a pose (checked: limits, FK round trip, collision via the IK/FK's <code>Code</code>).
<b>IK configuration</b>: the posture family it belongs to, named by answers to the arm's questions — UR5e: shoulder front/back · elbow up/down · wrist J5+/J5- → 8 configurations (<code>front/up/J5+</code>…). “Branch” is gone. The type is <code>IKConfigurations</code> + <code>ConfigurationAxis</code> (round 1's <code>IKBranches</code>, renamed; <code>round2/ik_configurations.py</code>). <code>src/</code> is gaining its own <code>IKConfigurations</code> + <code>ConfigurationQuestion</code> in parallel (another agent, uncommitted while this was built); none of the five depends on which one lands, only on a per-solution configuration index and its labels. Solutions per configuration in this result: {per_cfg}.</li>
<li><b>Workspaces are named by which orientations must work, never by a DOF count</b> (you: “how is hemisphere 5+ dof?” — right, it isn't): <code>full</code> every grid point analysed · <code>reachable</code> at least one orientation · <code>top_down</code> straight down at every roll (today's “four_dof”) · <code>hemisphere</code> every approach within 90° of down, every roll · <code>dexterous</code> every sampled orientation.</li>
<li><b>TCP</b> = whatever frame the FK/IK return; <b>collision</b> arrives as the IK/FK's <code>Code</code>, already folded into which IK solutions exist. None of the five proposals knows about either.</li>
</ul>

<h3>One numpy core, five surfaces</h3>
<pre>robot_reach                       robot_msgs (the seam)                            trajectory generation
ReachResult ──from_result()──▶  ChainReach arrays: positions (P,3) · orientations (O,3,3)  ──▶  P1 ws.hemisphere.inside(belt).sample_pairs()
 (FK/IK, Codes,                  kind_orientations · solution_pose (K,) · solution_q (K,dof)       P2 ws.sample(Q.hemisphere…, k=2)
  IK configuration tags)         solution_configuration (K,) · IKConfigurations                    P3 ws.sample(mask (P,O,C), k=2)
                                 pure numpy; no robot_reach import                                 P4 PoseSampler.pairs()   P5 table[…].sample_groups()</pre>
<p class="small">The UR5e result has {f['P']:,} grid points ({f['voxel_mm']:.0f} mm) × {f['O']} orientations × {f['C']} IK configurations, and {f['K']:,} checked IK solutions. The belt zone is 0.30–0.65 m in front, ±0.40 m across and 0.05–0.40 m up (the round-1 zone, given as a robot_msgs <code>SolidPrimitive.box</code> at a centre point).
The two-arm cell is the same UR5e twice, at y = ±{f['offset_m']:.3f} m (4 voxels each side). WHY a whole number of voxels: both arms then share one grid, so “both reach it” is a key intersection and not a nearest-point search. All four jobs ran in every proposal.</p>

<h3>The same four jobs, in each shape</h3>
<div class="scroll"><table class="small"><tr><th></th>{head}</tr>{side}</table></div>

<div class="grid" style="grid-template-columns:1fr">{''.join(cards)}</div>

<h3>Scoring</h3>
<details><summary class='muted small'>How the pick was made: criteria fixed from your asks before scoring, scores 1–5</summary><ul class='small'>{crit_why}</ul></details>
<table><tr><th>Proposal</th>{crit_head}<th class='num'>Weighted</th></tr>{crit_rows}</table>

<h3>Side by side, and against what exists</h3>
<div class="scroll"><table class="small">
<tr><th></th><th>Shape</th><th>robot_msgs type(s)</th><th>Holds for this UR5e</th><th>IK configuration filter</th><th>Several chains</th><th>Can't</th></tr>
<tr class="pick"><td>P1 Attribute namespace</td><td>properties → lazy regions</td><td><code>ReachableWorkspace</code>, <code>Cell</code></td><td>{mb('sparse (P1, P2)')} (sparse)</td><td><code>.only(elbow="up")</code>, re-judged</td><td><code>cell.left.hemisphere &amp; …</code></td><td>a custom cone without a new method</td></tr>
<tr><td>P2 Query builder</td><td>query value + workspace</td><td><code>WorkspaceQuery</code>, <code>ReachableWorkspace</code></td><td>{mb('sparse (P1, P2)')} + a few hundred bytes of JSON</td><td><code>.configuration(elbow="up")</code>, in the JSON</td><td><code>.on("left")</code>, <code>sample_shared</code></td><td>read as one call</td></tr>
<tr><td>P3 Numpy masks</td><td>dense bool cube</td><td><code>ReachableWorkspace</code> (+ <code>ok</code>)</td><td>{mb('sparse (P1, P2)')} + {by['dense cube ok (P3, extra)'] / 1e6:.1f} MB cube</td><td><code>ws.ok &amp; ws.configuration(…)</code></td><td>chain axis on a shared grid</td><td>stay small on fine grids</td></tr>
<tr><td>P4 Sampler first</td><td>pre-cut draw set</td><td><code>PoseSampler</code>, <code>CoupledSampler</code></td><td>3.0 MB for the belt cut · 82 MB for all reachable</td><td>at the producer only</td><td><code>CoupledSampler(couple=…)</code></td><td>full, contains, a new zone</td></tr>
<tr><td>P5 Table of IK solutions</td><td>one row per IK solution</td><td><code>SolutionTable</code></td><td>{mb('table (P5)')} (strings per row)</td><td><code>t.elbow == "up"</code>, then <code>complete()</code></td><td><code>chain</code> column</td><td>stay honest after a filter without <code>complete()</code></td></tr>
<tr class="muted"><td>Today: G2 (round 1 pick)</td><td>one kind per object</td><td><code>ReachableWorkspace(kind=…)</code></td><td>sparse, one kind</td><td><code>types=UR5E_BRANCHES.allowed(…)</code></td><td>not designed</td><td>switch kind without re-cutting; “branch” wording</td></tr>
<tr class="muted"><td>bam_reach v1 (Jul 2026)</td><td>attribute namespace</td><td>none (bam_reach classes)</td><td>dense (P, O, J) first solution</td><td>none (first solution only)</td><td>none</td><td>pairs, zones, configurations; consumers import bam_reach</td></tr>
<tr class="muted"><td>robot_trajectories today</td><td>duck-typed <code>ReachMapLike</code></td><td>none</td><td><code>joint_solutions</code> (P, O, J), <code>valid_mask</code> (P, O)</td><td>none</td><td>none</td><td>Cartesian zones; it thinks in joints</td></tr>
</table></div>
<p class="small"><b>robot_trajectories keeps working under any of the five</b>: the core's <code>reach_map_like(reach, allowed)</code> gives its duck type plus v1's <code>reach_scores</code>. Run here with elbow up only: <code>joint_solutions</code> {tuple(rml['joint_solutions'])}, <code>valid_mask</code> {tuple(rml['valid_mask'])}, <code>reach_scores</code> {tuple(rml['reach_scores'])}, {rml['valid']:,} valid (position, orientation) pairs. In P1 that is <code>region.as_reach_map_like()</code>, cut to the region (run below).</p>

<h3>Recommendation</h3>
<ol class="small">
<li><b>P1's surface is the robot_msgs type</b>: <code>ReachableWorkspace</code> with <code>full / reachable / top_down / hemisphere / dexterous</code> as properties, each a lazy region with <code>inside</code>, <code>only</code>, <code>contains</code>, <code>sample</code>, <code>sample_pairs</code> and <code>sample_waypoints</code>. It scored highest because the three happy paths are your sentence, and <code>.only()</code> re-judges membership, so a filter cannot overclaim. It is also v1's shape, so there is nothing to relearn.</li>
<li><b>Borrow P2 for P1's one gap</b>: a region's state (kind, zones, configurations, cone) <i>is</i> a <code>WorkspaceQuery</code>. So <code>region.spec()</code> (JSON) logs exactly what a campaign drew from, <code>ws.region(spec)</code> would re-open it on another arm (not built: the spec's zones are names here, not yet round-trippable shapes), and <code>ws.reachable.within(30)</code> is the custom cone without a sixth attribute.</li>
<li><b>Keep P4 as an export, not as the type</b>: <code>region.to_sampler().save(…)</code> writes a 3 MB file for a training loop that only draws.</li>
<li><b>Leave P3 and P5 as implementation details</b>: masks are how the core computes, and a table is a notebook view (<code>region.to_table()</code>). As public types, P3 is the longest to type and is dense, and P5's flags go stale.</li>
<li><b>Several chains</b>: <code>Cell</code> maps chain name → <code>ReachableWorkspace</code> in one frame, and <code>&amp;</code> on regions means “every chain reaches it”. A one-arm workspace is never wrapped, so single-chain code never changes.</li>
</ol>
<p class="small"><b>The borrowed pieces, added to the P1 prototype and run</b> (<code>within</code>, <code>spec</code>, <code>to_sampler</code>, <code>as_reach_map_like</code>; <code>over_belt</code> is job 2's region):</p>
{extras}
<p class="muted small">Found while running: the same-configuration A→B pairs still move one joint 70–180° (named in each line: mostly J6, the roll, and J4/J5). The roll and approach of each waypoint are drawn independently, so “same configuration” keeps the arm's posture family but not its wrist. If the generator wants gentle pairs, the next knob is a policy on the draw (e.g. <code>max_orientation_change_deg</code>) — not built. As in round 1, nothing checks the path between waypoints; only the waypoints themselves are guaranteed.</p>
<p class="muted small">Not run: <code>SolutionTable.to_pandas()</code> (pandas is not installed in robot_reach's environment; <code>head()</code> printed the same rows). P4's rotations are stored as float32, hence its 1e-2° FK residual against the others' 1e-5°. The two-arm cell is the same UR5e placed twice, not a real second chain. Nothing in <code>src/</code> changed; robot_msgs was not edited (the dataclasses are proposals).</p>
"""

# robot_reach — working notes for Claude Code

BAM Reach v2 (project note: ~/zach_brain/PROJECT - BAM Reach.md). Read README.md first.

- Runtime deps stay numpy + robot_msgs. viser / matplotlib are extras, imported lazily.
- The `Arm` contract (arm.py docstring) is the API: batched FK/IK, every IK branch, a `Code` each. Changing it is a barrier: land it alone.
- The library re-checks every IK claim (limits, FK round trip, collision, torque, singular). Never trust a solver's OK.
- Saved results are pickle-free (`*.reach.npz`, `result.json`); bump the schema string when the layout changes.
- overlay.py must keep working on viser 1.0.16 AND 1.1.1: no viser imports, only calls present in both.
- Angles shown to humans in degrees; data in radians, SI.
- Verify viewer changes with the headless-viser skill (real browser), not only tests/test_viewer.py.
- Host: when the robot loop is live, ≤2 worker processes, `taskset -c 2-5,16,17`.
- Rest poses (`rest.py`, `rest_viewer.py`, `rest_plots.py`): footprint / clearance / holding power over a `Scene`. An arm's FK may keep
  state (bam_fb_v2's MuJoCo data), so anything measuring from two threads goes through the viewer's `arm_lock`.
- Arm-specific adapters (bam_fb_v2: bam_ws `docs/named_poses/rest_explorer/`) live with the arm, never here.

# Autonomous Systems Lab - Team 1

This repository documents two pieces of work: an SVM-based line-following
controller trained on annotated CARLA footage, and a truck-platooning
communication protocol specified in SysML/UML and formally verified in
UPPAAL.

## SVM Line Following

`detect_line(self, image) -> float | None` takes a camera frame and returns
a continuous steering value in `[-1.0, 1.0]`, or `None` if the line is not
detected. Steering is treated as regression rather than classification, since
a classifier would force output into discrete buckets and produce jerky
corrections at class boundaries.

**Data and annotation.** Training frames were recorded as ROS bags while
driving through CARLA and extracted at regular intervals, covering straight
sections, gradual bends, and sharp turns with unfiltered, realistic
backgrounds. Each frame was manually annotated by drawing the intended lane
path from the vehicle position toward the expected lane centre further
ahead, using a consistent starting position and look-ahead distance.

**Feature extraction.** Each frame is reduced to a fixed-size feature vector:
HSV saturation boosted to 150% for lighting robustness, a green colour
threshold, a region-of-interest crop to the bottom 50% / centre 70% of the
frame, nearest-to-vehicle blob selection to resolve forks or multiple line
segments, then resize to 64x64 and flatten to a 4096-dim binary mask.

**Label generation.** Each annotated path is converted to a signed steering
value from a heading term (top-to-bottom horizontal displacement of the
path) and an offset term (displacement of the path's base from frame
centre), combined, scaled, and clipped to `[-1.0, 1.0]`.

**Model.** A `scikit-learn` pipeline of `StandardScaler` + `SVR` (RBF
kernel, `C=1.0`, `epsilon=0.1`), stored at `models/svm_line_follower.pkl`.
The epsilon-insensitive loss lets the model ignore pixel-level noise in the
mask instead of fitting to it. At inference, a five-frame moving average
smooths the raw prediction before it is used for steering; a discrete label
(Hard/Slight Left, Straight, Slight/Hard Right) is derived from the smoothed
value for logging only and plays no part in control.

## Truck Platooning

Independent trucks form a synchronized convoy over V2V communication to cut
drag, fuel use, and driver workload. The protocol is specified across three
scenarios, each modelled with SysML requirements and UML state machines and
then formalised as timed automata and verified in UPPAAL.

### Scenario 1: Coupling

A lone truck requests to join an already-moving convoy. It sends a join
request, is authorised via a cloud/NOC check, establishes a bounded-latency
V2V link, and passes a safety check on the inter-vehicle gap, lane
conditions, and blind spots. On success it synchronises speed with the lead
truck via ACC at 50 Hz over three rounds, then merges into its slot and
becomes an active platoon member. Failing the safety check, or a timeout or
link loss at any earlier step, aborts the truck back to independent driving.

### Scenario 2: Decoupling

A platoon member departs safely without endangering itself, the remaining
platoon, or surrounding traffic. The truck sends a leave request and waits
for approval, the following vehicles create a departure gap of at least
50 m within 5 seconds, a safety verification checks blind spots and lane
conditions, and the truck then exits its slot and changes lane under active
collision avoidance. The remaining trucks reorganise their positions,
spacing, and speed once the departure is complete. Two `LeavingTruck`
instances are modelled so that concurrent departure requests are handled
correctly, serialised one at a time by the lead truck.

### Scenario 3: Communication Failure and Safe Fallback

The lead truck emits a heartbeat every 100 ms; a watchdog resets its timeout
on every heartbeat received. If no heartbeat arrives for 300 ms, the
watchdog declares the link lost, broadcasts a failure event, and the
affected truck returns to independent driving. Safe fallback must be
reached within 200 ms: the onboard controller maintains lane position,
monitors distance, enlarges the gap, and triggers emergency braking if the
minimum safe distance is violated. Recovery is conditional: if the
heartbeat returns, the watchdog resumes monitoring, but the truck does not
automatically re-couple and must run a new join procedure.

### Verification

All three scenarios are combined into one UPPAAL model (joining truck, lead
truck, two leaving trucks, and a watchdog) and checked with the symbolic
model checker. Verified properties include:

- **Deadlock freedom** across the full composed system, including under
  contention between two trucks leaving at once.
- **Reachability** of coupling, decoupling, and emergency braking, so none
  of these paths are dead code.
- **Bounded timing**: join/leave approval within 100 ms, fallback within
  200 ms, link loss declared within 300 ms.
- **Resolution**: every safety-verification decision resolves to either a
  completed manoeuvre or a safe return to the previous state, never a stuck
  state.
- **Conditional recovery**: the truck stays in fallback indefinitely if the
  link does not return, rather than falsely assuming reconnection.

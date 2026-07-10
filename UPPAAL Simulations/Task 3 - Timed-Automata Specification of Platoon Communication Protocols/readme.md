# Task 3: Communication Protocols (Joining & Leaving)

This task models how trucks communicate during two key platoon scenarios: a truck **joining** the platoon and a truck **leaving** it. Both scenarios are built and verified as timed automata in UPPAAL.

---

## System Components

The model consists of three templates that work together as a single cohesive system:

| Component | Description |
| :--- | :--- |
| **`JoiningTruck`** | Sends a join request, waits for approval, executes a safety check, synchronizes speed, and settles into the platoon. |
| **`LeavingTruck`** | Sends a leave request, opens up a safe gap, verifies the exit is safe, and departs. |
| **`LeadTruck`** | The central coordinator. It handles one request at a time, approves joins and leaves, and emits a periodic heartbeat while a platoon is active. |

> **Note on Communication Flow:** The trucks never communicate directly with each other. All traffic flows through the leader using synchronized channels (e.g., `join_req`/`join_appr`, `leave_req`/`leave_appr`).

---

## Timing Assumptions

The critical timing constraints in this model are sourced from:
* **Internal SysML Scenarios:** Strict operational bounds, such as maintaining notifications under **100 ms** and a **50 Hz** speed synchronization rate.
* **Kamali et al., 2016:** Physical maneuver durations (e.g., gap creation and spacing). 

*Note: Any timing parameters lacking a specific source are flagged as modeling assumptions in the UPPAAL declarations.*

---

## A Note on Verification Results

When running the verifier, some liveness queries (e.g., `WaitApproval --> PlatoonActive`) will return as **not satisfied**. 

**This is intentional and not a bug.** It means the protocol is allowed to abort safely. If an insertion is unsafe or a gap cannot be created in time, the system will not force completion. System completion is *possible*, but not guaranteed at the expense of safety. This represents the correct, expected behavior of the automated protocol.

---

## Scope: No Communication Failure

This task models strictly the **normal communication flow**—meaning all messages assume the V2V (Vehicle-to-Vehicle) link is functioning perfectly. 

Communication failure, link loss, heartbeat timeouts, and the fallback to independent driving are **NOT** part of Task 3. Those edge cases and fallback safety mechanisms are handled in **Task 4**, which introduces the watchdog and robustness behaviors on top of this baseline model.

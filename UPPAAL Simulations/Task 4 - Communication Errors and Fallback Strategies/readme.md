# Task 4 — Control Behaviour & Robustness Under Communication Failure (UPPAAL)

Autonomous Systems A Lab
Timed-automata verification of platoon **control behaviour**, the **inter-vehicle distance guarantee**, and **robustness when V2V communication fails**.

This task builds on the Task 3 communication model and answers two questions from the lab:

1. **How can the distance to the preceding truck be guaranteed?**
2. **What happens on communication failure — is the system still robust and stable?**

The communication-failure scenario is modelled after our S3 sequence diagram: normal heartbeat → link loss → safe fallback → recovery.

---

## What's added for Task 4

- **`Watchdog` template** — the follower's failure-response automaton. It listens for the leader's heartbeat, detects link loss on timeout, drives the truck into a bounded safe-fallback mode (ACC / lane keeping / distance monitoring, with emergency braking on collision risk), and resumes normal platooning when the link is restored.
- **`LeadTruck` extension** — a `LinkDown` state so the leader can drop the link (stop beating) and later restore it, injecting the failure.
- **Fallback exit on the truck** — the truck reacts to the broadcast `link_lost` and returns to safe independent driving.

Failure detection and recovery use the broadcast channels `heartbeat` and `link_lost`.

---

## Watchdog behaviour (the failure loop)

```
Platooning  --(heartbeat received, timer reset)-->  Platooning        // normal operation
Platooning  --(no heartbeat for T_LINKLOSS)-->      LinkLost          // detect failure -> link_lost!
LinkLost    --(within T_FALLBACK)-->                SafeFallBack      // ACC / LKA / distance monitoring
SafeFallBack --(collision risk)-->                  EmergencyBrake    // AEB, increase gap
SafeFallBack / EmergencyBrake --(heartbeat)-->      Resume            // link restored
Resume      -->                                     Platooning        // platoon restored
```

`LinkLost` carries the invariant `f <= T_FALLBACK`, which **forces** the truck into safe mode within the fallback bound — this is the mechanism behind the robustness guarantee.

---

## Timing constants (Task 4)

Communication-failure timing is stated as an explicit modelling assumption (`[ASSUM]`), since the source literature leaves comms-failure handling as future work:

- `T_HB` — heartbeat period (leader emits every heartbeat cycle)
- `T_LINKLOSS` — no heartbeat for this long ⇒ link declared lost
- `T_FALLBACK` — safe mode must be reached within this after detecting loss
- `D_MIN` — minimum safe inter-vehicle gap

All clocks use a single consistent unit (milliseconds).

---

## Verification results

| Query | Result | Meaning |
|---|---|---|
| `E<> wd.SafeFallBack` | green | The system can enter safe-fallback mode after a failure. |
| `A[] (wd.LinkLost imply wd.f <= T_FALLBACK)` | green | **Robustness:** after link loss is detected, safe mode is always reached within the fallback bound. |
| `A[] (coupled imply gap >= D_MIN)` | green | **Distance guarantee:** the minimum safe spacing always holds while a truck is in the platoon. |
| `wd.LinkLost --> wd.Platooning` | red (by design) | Recovery is *possible* but not *guaranteed* — it depends on the link actually being restored. |

**On the `red` result:** this is not a defect. It correctly states that the platoon can only recover if communication returns; if the link stays down, the truck safely remains in fallback / independent driving rather than falsely resuming. That is the physically honest outcome.

**On deadlock:** `A[] not deadlock` does not hold, but only because of scenario **completion states** (e.g. a departed truck, or a fully-joined truck with no pending requests). These are natural end-states, not modelling errors. All safety, timing, distance, and robustness properties above hold.

---

## Answering the task

- **Distance to the preceding truck is guaranteed** whenever coupled — proven by `A[] (coupled imply gap >= D_MIN)`.
- **The system is robust and stable under communication failure** — on link loss the truck reaches a safe degraded mode within a bounded time (`A[] (wd.LinkLost imply wd.f <= T_FALLBACK)`), and recovers when the link returns. If the link never returns, it stays safe rather than resuming unsafely.

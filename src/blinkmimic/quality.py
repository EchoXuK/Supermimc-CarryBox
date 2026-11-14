"""Episode-local motion quality at the control rate; no differences across resets."""

import numpy as np

GAIT_METRICS_VERSION = 2


def contact_events(height, dt=.02, on=.008, off=.015, confirm=2):
    """Geometric contact with hysteresis and confirmation; never invent an S0 landing."""
    height = np.asarray(height)
    state = bool(height[0] < on)
    contacts = np.empty(len(height), dtype=bool)
    pending, first = 0, 0
    for t, z in enumerate(height):
        proposed = z < (off if state else on)
        if proposed != state:
            if pending == 0:
                first = t
            pending += 1
            if pending >= confirm:
                state, pending = bool(proposed), 0
                contacts[first:t + 1] = state
        else:
            pending = 0
        contacts[t] = state
    touchdown = np.flatnonzero(contacts[1:] & ~contacts[:-1]) + 1
    liftoff = np.flatnonzero(~contacts[1:] & contacts[:-1]) + 1
    return contacts, touchdown, liftoff


def match_events(actual, reference, tolerance=6):
    """Ordered, maximum-cardinality matching, then minimum total timing error."""
    actual, reference = list(actual), list(reference)
    n, m = len(actual), len(reference)
    score = [[(0, 0)] * (m + 1) for _ in range(n + 1)]
    move = np.zeros((n + 1, m + 1), dtype=np.uint8)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            options = [(score[i - 1][j], 1), (score[i][j - 1], 2)]
            d = abs(actual[i - 1] - reference[j - 1])
            if d <= tolerance:
                k, cost = score[i - 1][j - 1]
                options.append(((k + 1, cost - d), 3))
            score[i][j], move[i, j] = max(options, key=lambda x: x[0])
    pairs, i, j = [], n, m
    while i and j:
        direction = move[i, j]
        if direction == 3:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif direction == 1:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def gait_quality(actual, reference, dt=.02):
    """Episode-local sole geometry, including heel/toe rocking below the old 4 cm gate.

    Input is (frames, left/right, four heel/toe pads, xyz). Reference may contain
    the remainder after failure; metrics compare only the actual valid prefix.
    """
    actual, reference = np.asarray(actual), np.asarray(reference)
    T = min(len(actual), len(reference))
    if T < 5:
        return None
    trim, tail = min(round(.5 / dt), (T - 1) // 4), min(round(.24 / dt), (T - 1) // 8)
    stop = T - tail
    floor = np.full(2, np.quantile(reference[..., 2].min(-1), .05))
    a, r = actual[:T], reference[:T]
    reference_whole_contact = [contact_events(r[:, j, :, 2].min(-1) - floor[j], dt)[0] for j in range(2)]
    regions = {"whole": (slice(None), .008, .015), "heel": (slice(0, 2), .005, .010),
               "toe": (slice(2, 4), .005, .010)}
    details, extras, whole_actual, whole_reference = {}, [], [], []
    missing = 0
    for region, (pads, on, off) in regions.items():
        extra_count, missed_count, errors = 0, 0, []
        for foot in range(2):
            ah = a[:, foot, pads, 2].min(-1) - floor[foot]
            rh = r[:, foot, pads, 2].min(-1) - floor[foot]
            ac, at, al = contact_events(ah, dt, on, off)
            rc, rt, rl = contact_events(rh, dt, on, off)
            at, rt = at[(at >= trim) & (at < stop)], rt[(rt >= trim) & (rt < stop)]
            pairs = match_events(at, rt, round(.12 / dt))
            ai, ri = {i for i, _ in pairs}, {j for _, j in pairs}
            for i, t in enumerate(at):
                if i not in ai:
                    extras.append(dict(frame=int(t), foot=foot, region=region,
                                       reference_stance=bool(reference_whole_contact[foot][t])))
            extra_count += len(at) - len(ai)
            missed_count += len(rt) - len(ri)
            errors.extend(abs(int(at[i]) - int(rt[j])) * dt for i, j in pairs)
            if region == "whole":
                whole_actual.extend((int(t), foot) for t in at)
                whole_reference.extend((int(t), foot) for t in rt)
                missing += len(rt) - len(ri)
        details[region] = dict(extra_touchdowns=extra_count, missed_touchdowns=missed_count,
                               matched_timing_mae_s=float(np.mean(errors)) if errors else None)
    # A full-foot landing can also land a heel/toe pad; count one physical event
    # within 80 ms per foot, while retaining region-level diagnostics.
    unique = []
    for event in sorted(extras, key=lambda x: (x["frame"], x["foot"])):
        previous = next((v for v in reversed(unique) if v["foot"] == event["foot"]), None)
        if previous is not None and event["frame"] - previous["frame"] <= round(.08 / dt):
            previous["regions"].append(event["region"])
        else:
            unique.append(dict(frame=event["frame"], foot=event["foot"], regions=[event["region"]],
                               reference_stance=event["reference_stance"]))
    def repeated(events):
        ordered = sorted(events)
        return sum(a[1] == b[1] and a[0] != b[0] for a, b in zip(ordered, ordered[1:]))
    duration = (stop - trim) * dt
    av, rv = np.gradient(a, dt, axis=0), np.gradient(r, dt, axis=0)
    rh = r[..., 2].min(-1) - floor
    stance = np.stack([contact_events(rh[:, j], dt)[0] for j in range(2)], -1)
    stance[:trim], stance[stop:] = False, False
    speed = np.linalg.norm(av[..., :2], axis=-1).mean(-1)
    refspeed = np.linalg.norm(rv[..., :2], axis=-1).mean(-1)
    return dict(version=GAIT_METRICS_VERSION, seconds=duration, extra_recontacts=len(unique),
                extra_recontacts_per_minute=len(unique) * 60 / duration,
                whole_extra_touchdowns=details["whole"]["extra_touchdowns"],
                heel_extra_touchdowns=details["heel"]["extra_touchdowns"],
                toe_extra_touchdowns=details["toe"]["extra_touchdowns"],
                missed_steps=missing, repeated_same_foot=repeated(whole_actual),
                reference_repeated_same_foot=repeated(whole_reference),
                swing_interruptions=sum(not v["reference_stance"] for v in unique),
                stance_speed_m_s=float(speed[stance].mean()) if stance.any() else 0.,
                reference_stance_speed_m_s=float(refspeed[stance].mean()) if stance.any() else 0.,
                regions=details, extra_events=unique,
                actual_touchdowns=[dict(frame=t, foot=j) for t, j in sorted(whole_actual)],
                reference_touchdowns=[dict(frame=t, foot=j) for t, j in sorted(whole_reference)])


def mean_gait(values):
    valid = [v for v in values if v is not None]
    if not valid:
        return dict(clips=0)
    keys = [k for k, v in valid[0].items() if isinstance(v, (float, int)) and k != "version"]
    return dict(clips=len(valid), **{k: float(np.mean([v[k] for v in valid])) for k in keys})


def _joint_metrics(q, dt):
    if len(q) < 4:
        return None
    v = np.diff(q, axis=0) / dt
    acc = np.diff(q, n=2, axis=0) / dt**2
    centered = v - v.mean(0, keepdims=True)
    power = np.abs(np.fft.rfft(centered * np.hanning(len(v))[:, None], axis=0))**2
    frequencies = np.fft.rfftfreq(len(v), dt)
    return dict(vel_rms=float(np.sqrt(np.mean(v**2))), acc_rms=float(np.sqrt(np.mean(acc**2))),
                hf6_fraction=float(power[frequencies >= 6.].sum() / max(float(power.sum()), 1e-30)))


def episode_quality(q, reference_q, actions, action_scale, default_q, dt=.02):
    """q/reference_q include S0 and terminal state; actions have one row per transition.

    Exclude min(0.5 s, quarter episode) of reset settling. Targets align with q[1:].
    The 6 Hz band is a comparison metric, not a claim that every fast movement is bad.
    """
    q, reference_q, actions = map(np.asarray, (q, reference_q, actions))
    T = len(actions)
    if len(q) != T + 1 or reference_q.shape != q.shape:
        raise ValueError("Quality needs aligned reference and actual states, including endpoints")
    trim = min(round(.5 / dt), T // 4)
    target = np.asarray(default_q) + np.asarray(action_scale) * actions
    series = dict(actual=q[trim + 1:], reference=reference_q[trim + 1:], target=target[trim:],
                  residual=(target - reference_q[1:])[trim:])
    result = {name: _joint_metrics(x, dt) for name, x in series.items()}
    return result if all(v is not None for v in result.values()) else None


def mean_quality(values):
    valid = [v for v in values if v is not None]
    if not valid:
        return dict(clips=0)
    return dict(clips=len(valid), **{key: {metric: float(np.mean([v[key][metric] for v in valid]))
                                         for metric in valid[0][key]} for key in valid[0]})


def foot_quality(actual, reference, reference_velocity, dt=.02):
    """Foot slip and extra lifts while the demonstration requires stance (ankle origins)."""
    T = len(actual)
    if T < 4:
        return None
    floor = np.quantile(reference[..., 2], .05, axis=0)
    stance = ((reference[..., 2] < floor + .015)
              & (np.linalg.norm(reference_velocity[..., :2], axis=-1) < .5))[:T]
    trim = min(round(.5 / dt), (T - 1) // 4)
    stance[:trim] = False
    actual_speed = np.linalg.norm(np.gradient(actual, dt, axis=0)[..., :2], axis=-1)
    ref_speed = np.linalg.norm(reference_velocity[:T, ..., :2], axis=-1)
    raised = actual[..., 2] > reference[:T, ..., 2] + .04
    extra = raised[1:] & ~raised[:-1] & stance[1:] & stance[:-1]
    count = max(int(stance.sum()), 1)
    return dict(stance_speed_m_s=float(actual_speed[stance].sum() / count),
                reference_stance_speed_m_s=float(ref_speed[stance].sum() / count),
                extra_lift_count=int(extra.sum()), reference_stance_seconds=float(stance.sum() * dt),
                raised_stance_fraction=float((raised & stance).sum() / count))

"""Episode-local motion quality at the control rate; no differences across resets."""

import numpy as np


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

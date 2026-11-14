"""Regression cases for full, clipped, and non-calibration post-release arm motions."""
import unittest

import numpy as np

from clean_box_carry_terminal_init import analyze, recovery_preserving_end, recovery_evidence, trim_payload


def example(n=260, release=100):
    a = np.zeros((n, 15))
    a[:, :2] = .25
    a[:, 2] = .42
    a[:, 3] = -.12
    a[:, 4] = .74
    a[:, 5] = 5.
    a[:, 8] = .78
    a[:, 11] = .2
    a[:release, 12:14] = 1
    a[:, 14] = np.arange(n)*1.8
    row = dict(clip_npz="example.npz", stage="03_lower_place", variant="augmented", split="test", capture_id="example")
    return a, row


def raise_arms(a, start=190, end=240, peak=1.6):
    a[start:end, :2] = np.linspace([.25, .25], [peak, peak], end-start)
    a[end:, :2] = peak
    a[start:end, 2] = np.linspace(.42, .42+.5*(peak-.25), end-start)
    a[end:, 2] = .42+.5*(peak-.25)
    a[start:end, 3] = np.linspace(-.12, .25, end-start)
    a[end:, 3] = .25


class TerminalInitTests(unittest.TestCase):
    def test_cuts_at_the_start_of_reinit_not_when_arms_reach_horizontal(self):
        a, row = example()
        raise_arms(a)
        result = analyze(a, row)
        self.assertLessEqual(result["keep_end"], 190)
        self.assertGreater(result["keep_end"], 180)
        self.assertGreater(result["keep_end"], result["last_contact"])

    def test_truncated_tpose_is_caught_below_old_point_nine_threshold(self):
        a, row = example(n=224)
        raise_arms(a, start=190, end=222, peak=.7)
        result = analyze(a, row)
        self.assertLess(result["keep_end"], 195)
        self.assertGreater(result["keep_end"], 180)

    def test_putting_down_box_and_returning_hands_keeps_the_whole_clip(self):
        a, row = example()
        a[100:160, :2] = np.linspace([.4, .4], [.25, .25], 60)
        a[100:160, 2] = np.linspace(.55, .42, 60)
        self.assertEqual(analyze(a, row)["keep_end"], len(a))

    def test_unilateral_arm_swing_is_not_calibration(self):
        a, row = example()
        raise_arms(a)
        a[:, 1] = .25
        self.assertEqual(analyze(a, row)["keep_end"], len(a))

    def test_wide_grasp_while_in_contact_is_preserved(self):
        a, row = example()
        raise_arms(a)
        a[:, 12:14] = 1
        self.assertEqual(analyze(a, row)["keep_end"], len(a))

    def test_motion_with_box_moving_again_is_not_silent_terminal_trim(self):
        a, row = example()
        raise_arms(a)
        a[180:, 9] = np.linspace(0., .3, len(a)-180)
        self.assertEqual(analyze(a, row)["keep_end"], len(a))

    def test_recovered_body_does_not_keep_any_extra_arm_ramp(self):
        a, row = example()
        raise_arms(a)
        onset = analyze(a, row)["keep_end"]
        end, detail = recovery_preserving_end(a, row, onset)
        self.assertEqual(end, onset)
        self.assertEqual(detail["overlap_retained_frames"], 0)

    def test_overlapping_rise_retains_useful_transition_before_wide_arms(self):
        a, row = example()
        raise_arms(a, start=190, end=250)
        a[:190, 4:6] = [.55, 50.]
        a[190:215, 4:6] = np.linspace([.55, 50.], [.74, 20.], 25)
        a[215:, 4:6] = [.74, 20.]
        onset = analyze(a, row)["keep_end"]
        end, detail = recovery_preserving_end(a, row, onset)
        self.assertGreater(end, onset)
        self.assertLess(a[end-1, 2], .72)
        self.assertLess(end, 230)

    def test_slow_rise_is_limited_before_spread_and_not_claimed_complete(self):
        a, row = example()
        raise_arms(a)
        a[:230, 4:6] = [.55, 55.]
        a[230:, 4:6] = [.74, 5.]
        onset = analyze(a, row)["keep_end"]
        end, detail = recovery_preserving_end(a, row, onset)
        self.assertEqual(detail["endpoint_reason"], "arm_clearance_limit_during_recovery")
        self.assertLess(a[end-1, 2], .72)
        self.assertLess(a[end-1, :2].min(), .75)
        evidence = recovery_evidence(a[:end], row)
        self.assertFalse(evidence["recovery_complete"])
        payload = dict(joint_pos=np.zeros((len(a),29)), joint_vel=np.ones((len(a),29)),
            source_frame_index=a[:,14], source_fps=np.array([90.]),
            source_end_frame_exclusive=np.array(467), recovery_complete=np.array(True),
            source_placement_end_frame_exclusive=np.array(180),
            source_recovery_stable_window_start=np.array(400), static_window=np.array([[100,120,140,160]]))
        record = dict(keep_end=end, init_onset_frame=onset, retained_recovery_evidence=evidence, **detail)
        out = trim_payload(payload, record)
        np.testing.assert_array_equal(out["joint_vel"], payload["joint_vel"][:end])
        np.testing.assert_array_equal(out["source_frame_index"], payload["source_frame_index"][:end])
        np.testing.assert_array_equal(out["static_window"], payload["static_window"])
        self.assertFalse(out["recovery_complete"].item())
        self.assertEqual(out["source_recovery_stable_window_start"].item(), -1)


if __name__ == "__main__":
    unittest.main()

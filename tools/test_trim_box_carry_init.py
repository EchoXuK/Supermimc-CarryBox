"""Behavior tests for init removal and preservation of actual object interaction."""
import unittest

import numpy as np

from trim_box_carry_init import detect_init, suffix


def motion(n=400):
    q = np.zeros((n, 29), dtype=np.float32)
    q[:, 15], q[:, 16] = .3, -.3
    body = np.zeros((n, 39, 3), dtype=np.float32)
    body[:, 0, 2] = .78
    body[:, 37, 0], body[:, 38, 0] = .225, -.225
    contact = np.zeros((n, 4, 1), dtype=np.uint8)
    contact[300:, 2:4] = 1
    box = np.tile([0., .5, .15], (n, 1)).astype(np.float32)
    return q, body, contact, box


def spread(q, body, start=80, plateau=100, release=150, end=170):
    angle = np.r_[np.linspace(.3, 1.6, plateau-start, endpoint=False),
                  np.full(release-plateau, 1.6), np.linspace(1.6, .3, end-release, endpoint=False)]
    q[start:end, 15], q[start:end, 16] = angle, -angle
    separation = .45 + .65*(angle-.3)/1.3
    body[start:end, 37, 0], body[start:end, 38, 0] = separation/2, -separation/2


class TrimInitTests(unittest.TestCase):
    def test_calibration_and_earlier_waiting_are_removed(self):
        q, body, contact, box = motion()
        spread(q, body)
        body[190:, 0, 1] = np.linspace(0., .5, len(q)-190)
        result = detect_init(q, body, contact, box)
        self.assertGreaterEqual(result["cut_frames"], 170)
        self.assertLess(result["cut_frames"], 190)
        self.assertLess(result["cut_frames"], result["first_contact_frame"])

    def test_contact_label_spike_during_tpose_does_not_hide_init(self):
        q, body, contact, box = motion()
        spread(q, body)
        contact[120:122, 2:4] = 1
        result = detect_init(q, body, contact, box)
        self.assertEqual(result["first_contact_frame"], 300)
        self.assertGreater(result["cut_frames"], 150)

    def test_clipped_initial_tpose_return_is_removed(self):
        q, body, contact, box = motion()
        a = np.linspace(1.38, .3, 25)
        q[:25, 15], q[:25, 16] = a, -a
        body[:25, 37, 0] = (.45+.65*(a-.3)/1.3)/2
        body[:25, 38, 0] = -body[:25, 37, 0]
        result = detect_init(q, body, contact, box)
        self.assertGreaterEqual(result["cut_frames"], 24)
        self.assertLess(result["cut_frames"], 45)

    def test_normal_reaching_without_init_keeps_all_frames(self):
        q, body, contact, box = motion()
        q[40:200, 11:13] = -.8
        self.assertEqual(detect_init(q, body, contact, box)["cut_frames"], 0)

    def test_single_arm_motion_is_not_calibration(self):
        q, body, contact, box = motion()
        spread(q, body)
        q[:, 16] = -.3
        self.assertEqual(detect_init(q, body, contact, box)["cut_frames"], 0)

    def test_post_release_spread_does_not_delete_placement(self):
        q, body, contact, box = motion()
        contact[:200, 2:4] = 1
        contact[200:] = 0
        spread(q, body, 240, 260, 320, 340)
        self.assertEqual(detect_init(q, body, contact, box)["cut_frames"], 0)

    def test_slicing_preserves_contacts_timing_and_last_pose(self):
        q, body, contact, box = motion()
        spread(q, body)
        cut = detect_init(q, body, contact, box)["cut_frames"]
        payload = dict(joint_pos=q, body_pos_w=body, contact_info=contact, object_pos_w=box,
                       source_frame_index=11+np.arange(len(q))*1.8,
                       source_start_frame=np.array(11, dtype=np.int64),
                       fps=np.array([50.], dtype=np.float32),
                       object_geom_quat_wxyz=np.array([1., 0., 0., 0.], dtype=np.float32),
                       source_carry_newdof_windows=np.array([[12, 40, 0, 2]]))
        cleaned = suffix(payload, cut)
        for key in ("joint_pos", "body_pos_w", "contact_info", "object_pos_w", "source_frame_index"):
            np.testing.assert_array_equal(cleaned[key], payload[key][cut:])
            self.assertEqual(cleaned[key].dtype, payload[key].dtype)
            np.testing.assert_array_equal(cleaned[key][-1], payload[key][-1])
        np.testing.assert_array_equal(cleaned["source_carry_newdof_windows"], payload["source_carry_newdof_windows"])
        self.assertEqual(float(cleaned["source_start_frame_exact"]), payload["source_frame_index"][cut])


if __name__ == "__main__":
    unittest.main()

"""Cropped-tail reset regressions, exercising the production sampler on CPU."""
import ast
from pathlib import Path
from types import MethodType, SimpleNamespace
import unittest

import torch

from blinkmimic.motion import MotionLib

ROOT=Path(__file__).resolve().parents[1]


def fixture(lengths=(33,52,102), minimum=32):
    motion=MotionLib.__new__(MotionLib)
    motion.device='cpu'
    motion.clip_len=torch.tensor(lengths)
    motion.clip_start=torch.cat((torch.zeros(1,dtype=torch.long),motion.clip_len.cumsum(0)[:-1]))
    motion.clip_end=motion.clip_start+motion.clip_len-1
    bins=[(c,int(motion.clip_start[c])+b) for c,length in enumerate(lengths) for b in range(0,length-1,50)]
    motion.bin_clip=torch.tensor([c for c,_ in bins])
    motion.bin_start=torch.tensor([f for _,f in bins])
    motion.num_bins=len(bins)
    counts=torch.bincount(motion.bin_clip).float()
    motion.bin_prior=1/(len(lengths)*counts[motion.bin_clip])
    env=SimpleNamespace(motion=motion,device='cpu',carry=True,min_reset_steps=minimum,
        bin_G=torch.zeros(len(bins)),G_max=19.,bin_w_avg=motion.bin_prior.clone(),
        fail_reset=0.,bin_fail=torch.zeros(len(bins)),bin_prev=torch.arange(len(bins)),
        focus_static=False,focus_q=None,focus_share=0.)
    if minimum>1:
        env.reset_bin_end,env.reset_bin_prior=motion.reset_sampling(minimum)
        env.bin_w_avg.copy_(env.reset_bin_prior)
    tree=ast.parse((ROOT/'src/blinkmimic/env.py').read_text())
    cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='TrackingSim')
    fn=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='sample_frames')
    ns={'torch':torch}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'production_sampler','exec'),ns)
    env.sample_frames=MethodType(ns['sample_frames'],env)
    return env


class ResetSamplingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)

    def test_shortest_33_frame_clip_keeps_32_transitions(self):
        env=fixture((33,))
        frames,bins=env.sample_frames(1000)
        self.assertTrue(bool((frames==0).all()))
        self.assertTrue(bool((bins==0).all()))
        self.assertEqual(int(env.motion.clip_end[0]-frames[0]),32)

    def test_no_sampled_reference_tail_is_shorter_than_requested(self):
        env=fixture()
        frames,bins=env.sample_frames(20000)
        end=env.motion.clip_end[env.motion.bin_clip[bins]]
        self.assertTrue(bool((end-frames>=32).all()))
        self.assertTrue(bool((frames>=env.motion.bin_start[bins]).all()))
        self.assertEqual(set(env.motion.bin_clip[bins].tolist()),{0,1,2})

    def test_time_prior_preserves_clip_and_stage_mass(self):
        env=fixture()
        m=env.motion
        mass=torch.zeros(3).index_add_(0,m.bin_clip,env.reset_bin_prior)
        torch.testing.assert_close(mass,torch.full((3,),1/3))
        # Last clip has 70 eligible starts: 50 in bin 0, 20 in bin 1.
        torch.testing.assert_close(env.reset_bin_prior[m.bin_clip==2],torch.tensor([50/210,20/210,0.]))
        self.assertEqual(float(env.reset_bin_prior.sum()),1.)

    def test_invalid_extreme_curriculum_cannot_underflow_valid_probabilities(self):
        env=fixture()
        invalid=env.reset_bin_prior==0
        env.bin_G[invalid]=-1e9
        env.bin_w_avg.copy_(invalid.float()/invalid.sum())
        frames,bins=env.sample_frames(10000)
        self.assertTrue(bool(torch.isfinite(env.bin_w_avg).all()))
        self.assertTrue(bool((env.reset_bin_prior[bins]>0).all()))
        self.assertTrue(bool((env.motion.clip_end[env.motion.bin_clip[bins]]-frames>=32).all()))

    def test_failure_and_focus_mixtures_obey_the_window(self):
        env=fixture()
        invalid=(env.reset_bin_prior==0).float()
        env.fail_reset=.8
        env.bin_fail.copy_(invalid)
        env.focus_q=invalid/invalid.sum()
        env.focus_share=.8
        _,bins=env.sample_frames(10000)
        self.assertTrue(bool((env.reset_bin_prior[bins]>0).all()))

    def test_all_invalid_static_focus_uses_the_valid_prior(self):
        env=fixture()
        invalid=(env.reset_bin_prior==0).float()
        env.focus_static=True
        env.focus_share_t=torch.tensor(1.)
        env.focus_q=invalid/invalid.sum()
        _,bins=env.sample_frames(10000,gate=torch.tensor(False))
        self.assertTrue(bool((env.reset_bin_prior[bins]>0).all()))

    def test_legacy_default_still_allows_a_one_step_tail(self):
        env=fixture((52,),minimum=1)
        env.focus_q=torch.tensor([0.,1.])
        env.focus_share=1.
        frames,bins=env.sample_frames(1000)
        self.assertTrue(bool((frames==50).all()))
        self.assertTrue(bool((bins==1).all()))

    def test_invalid_window_is_rejected(self):
        m=fixture().motion
        for minimum in (0,33,1000):
            with self.subTest(minimum=minimum),self.assertRaises(ValueError):
                m.reset_sampling(minimum)


if __name__=='__main__':
    unittest.main()

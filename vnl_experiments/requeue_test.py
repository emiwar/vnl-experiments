"""Tests for the requeue machinery.

Everything here is pure Python -- no jax, no env -- because the parts of a
preemptible run that are easy to get wrong are the bookkeeping ones: which
directory attempt N picks, whether the WandB id survives, and whether a signal is
actually noticed. The training-side behaviour (`stop_fn`, `initial_eval`, light
checkpoints) is covered by nnx-ppo's own tests.
"""

import json
import os
import signal
import tempfile
from pathlib import Path

from absl.testing import absltest

from vnl_experiments import requeue


class RunTokenTest(absltest.TestCase):
    """The token is the whole resume mechanism: it must be the same on attempt 2
    as on attempt 1, and different between two separate submissions."""

    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in (
            "SLURM_JOB_ID", "SLURM_JOBID", "SLURM_ARRAY_JOB_ID",
            "SLURM_ARRAY_TASK_ID", "SLURM_RESTART_COUNT")}
        for k in self._saved:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_plain_job(self):
        os.environ["SLURM_JOB_ID"] = "4821993"
        self.assertEqual(requeue.run_token(), "job4821993")

    def test_stable_across_a_requeue(self):
        """A requeued job keeps its id; only SLURM_RESTART_COUNT moves."""
        os.environ["SLURM_JOB_ID"] = "4821993"
        first = requeue.run_token()
        os.environ["SLURM_RESTART_COUNT"] = "3"
        self.assertEqual(requeue.run_token(), first)
        self.assertEqual(requeue.restart_count(), 3)

    def test_array_task(self):
        os.environ["SLURM_JOB_ID"] = "4821999"
        os.environ["SLURM_ARRAY_JOB_ID"] = "4900000"
        os.environ["SLURM_ARRAY_TASK_ID"] = "7"
        # The array ids win: they are what stays put when a task is requeued.
        self.assertEqual(requeue.run_token(), "job4900000_7")

    def test_off_cluster_is_unique_per_invocation(self):
        token = requeue.run_token()
        self.assertTrue(token.startswith("local-"), token)

    def test_restart_count_defaults_to_zero(self):
        self.assertEqual(requeue.restart_count(), 0)
        os.environ["SLURM_RESTART_COUNT"] = "not-a-number"
        self.assertEqual(requeue.restart_count(), 0)


class RunStateTest(absltest.TestCase):

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def test_absent_state_is_none(self):
        self.assertIsNone(requeue.RunState.load(self.dir))

    def test_round_trip(self):
        requeue.RunState(wandb_id="abc123", exp_name="run-job1").save(self.dir)
        loaded = requeue.RunState.load(self.dir)
        self.assertEqual(loaded.wandb_id, "abc123")
        self.assertEqual(loaded.exp_name, "run-job1")
        self.assertEqual(loaded.attempts, [])
        self.assertFalse(loaded.done)

    def test_attempts_accumulate_across_loads(self):
        state = requeue.RunState(wandb_id="abc123", exp_name="run-job1")
        state.record_attempt(self.dir, resumed_from_step=None)
        state.finish_attempt(self.dir, "preempted", stopped_at_step=1000)

        reloaded = requeue.RunState.load(self.dir)
        reloaded.record_attempt(self.dir, resumed_from_step=1000)
        again = requeue.RunState.load(self.dir)

        self.assertEqual(len(again.attempts), 2)
        self.assertEqual(again.attempts[0]["outcome"], "preempted")
        self.assertEqual(again.attempts[0]["stopped_at_step"], 1000)
        self.assertEqual(again.attempts[1]["attempt"], 2)
        self.assertEqual(again.attempts[1]["resumed_from_step"], 1000)

    def test_done_survives(self):
        state = requeue.RunState(wandb_id="abc123", exp_name="run-job1")
        state.done = True
        state.save(self.dir)
        self.assertTrue(requeue.RunState.load(self.dir).done)

    def test_truncated_file_is_treated_as_absent(self):
        """Better to start fresh than to strand the run on a bad write."""
        requeue.RunState.path(self.dir).write_text("{not json")
        self.assertIsNone(requeue.RunState.load(self.dir))

    def test_unknown_keys_are_ignored(self):
        """An older/newer state file must not crash this attempt."""
        requeue.RunState.path(self.dir).write_text(json.dumps(
            {"wandb_id": "abc", "exp_name": "r", "from_the_future": 1}))
        self.assertEqual(requeue.RunState.load(self.dir).wandb_id, "abc")

    def test_save_leaves_no_temp_file(self):
        requeue.RunState(wandb_id="abc", exp_name="r").save(self.dir)
        self.assertEqual([p.name for p in self.dir.iterdir()], ["run_state.json"])


class PreemptionWatcherTest(absltest.TestCase):

    def test_notices_the_signal(self):
        watcher = requeue.PreemptionWatcher(signals=(signal.SIGUSR2,))
        try:
            self.assertFalse(watcher(0))
            os.kill(os.getpid(), signal.SIGUSR2)
            self.assertTrue(watcher(0))
            self.assertEqual(watcher.signal_name, "SIGUSR2")
        finally:
            watcher.restore()

    def test_callback_fires_once(self):
        seen = []
        watcher = requeue.PreemptionWatcher(signals=(signal.SIGUSR2,),
                                            on_signal=seen.append)
        try:
            os.kill(os.getpid(), signal.SIGUSR2)
            os.kill(os.getpid(), signal.SIGUSR2)
            self.assertEqual(seen, [signal.SIGUSR2])
        finally:
            watcher.restore()

    def test_second_signal_hands_control_back(self):
        """Signalling twice must not leave you stuck waiting for an iteration."""
        marker = []
        signal.signal(signal.SIGUSR2, lambda *a: marker.append("original"))
        watcher = requeue.PreemptionWatcher(signals=(signal.SIGUSR2,))
        os.kill(os.getpid(), signal.SIGUSR2)   # noticed, handler still ours
        os.kill(os.getpid(), signal.SIGUSR2)   # "I meant it": handler restored
        self.assertEqual(marker, [])
        os.kill(os.getpid(), signal.SIGUSR2)   # now the original one runs
        self.assertEqual(marker, ["original"])
        signal.signal(signal.SIGUSR2, signal.SIG_DFL)

    def test_restore_puts_the_old_handler_back(self):
        marker = []
        signal.signal(signal.SIGUSR2, lambda *a: marker.append("original"))
        watcher = requeue.PreemptionWatcher(signals=(signal.SIGUSR2,))
        watcher.restore()
        os.kill(os.getpid(), signal.SIGUSR2)
        self.assertEqual(marker, ["original"])
        self.assertFalse(watcher.triggered)
        signal.signal(signal.SIGUSR2, signal.SIG_DFL)

    def test_default_signals_include_sigterm(self):
        """SIGTERM is what Slurm actually sends when it preempts a job."""
        self.assertIn(signal.SIGTERM, requeue.PREEMPTION_SIGNALS)


class RewindNeededTest(absltest.TestCase):
    """Whether the WandB history runs past the checkpoint being restored.

    It does not after a graceful preemption -- ``stop_fn`` checkpoints at the
    exact step it stopped on -- and a plain resume then produces a seamless
    curve. Only a hard kill leaves logged steps ahead of the last checkpoint.
    """

    def _state(self, attempts):
        s = requeue.RunState(wandb_id="abc", exp_name="r")
        s.attempts = attempts
        return s

    def test_first_attempt_never_rewinds(self):
        self.assertFalse(requeue.rewind_needed(self._state([]), None))

    def test_graceful_preemption_needs_no_rewind(self):
        """Saved at 139520 and resuming at 139520: nothing logged past it."""
        state = self._state([
            {"attempt": 1, "outcome": "preempted", "stopped_at_step": 139520},
            {"attempt": 2},  # the current attempt, appended before wandb.init
        ])
        self.assertEqual(requeue.last_logged_step(state), 139520)
        self.assertFalse(requeue.rewind_needed(state, 139520))

    def test_hard_kill_with_no_recorded_ending_rewinds(self):
        state = self._state([{"attempt": 1}, {"attempt": 2}])
        self.assertIsNone(requeue.last_logged_step(state))
        self.assertTrue(requeue.rewind_needed(state, 100000))

    def test_history_ahead_of_the_checkpoint_rewinds(self):
        state = self._state([
            {"attempt": 1, "outcome": "preempted", "stopped_at_step": 150000},
            {"attempt": 2},
        ])
        self.assertTrue(requeue.rewind_needed(state, 100000))

    def test_finished_step_counts_as_logged(self):
        state = self._state([
            {"attempt": 1, "outcome": "finished", "final_step": 400640},
            {"attempt": 2},
        ])
        self.assertEqual(requeue.last_logged_step(state), 400640)


class WandbConfigTest(absltest.TestCase):

    def test_provenance_lists_every_resume(self):
        state = requeue.RunState(wandb_id="abc", exp_name="r")
        state.attempts = [
            {"attempt": 1, "resumed_from_step": None},
            {"attempt": 2, "resumed_from_step": 100},
            {"attempt": 3, "resumed_from_step": 250},
        ]
        cfg = requeue.wandb_requeue_config(state, "/ckpt/r", 250, attempt=3)
        self.assertEqual(cfg["requeue"]["attempt"], 3)
        self.assertEqual(cfg["requeue"]["resumed_from_step"], 250)
        self.assertEqual(cfg["requeue"]["resumed_steps"], [100, 250])
        self.assertEqual(cfg["requeue"]["checkpoint_dir"], "/ckpt/r")


if __name__ == "__main__":
    absltest.main()

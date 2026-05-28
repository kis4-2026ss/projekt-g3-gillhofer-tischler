import unittest
from src.orchestrator.nodes.executor import ready_subtasks, prereq_context
from src.orchestrator.state import SubTask


def _task(task_id, deps=None, status="assigned", result=None):
    return SubTask(
        id=task_id,
        description=f"task {task_id}",
        dependencies=deps or [],
        status=status,
        result=result,
    )


def _as_dict(*tasks):
    return {t.id: t for t in tasks}


class TestReadySubtasks(unittest.TestCase):

    def test_independent_tasks_are_all_ready(self):
        subs = _as_dict(_task("t1"), _task("t2"))
        self.assertEqual({t.id for t in ready_subtasks(subs)}, {"t1", "t2"})

    def test_dependent_task_waits_for_prerequisite(self):
        subs = _as_dict(_task("t1"), _task("t2", ["t1"]))
        self.assertEqual([t.id for t in ready_subtasks(subs)], ["t1"])

    def test_dependent_task_becomes_ready_when_prereq_completes(self):
        subs = _as_dict(_task("t1", status="completed", result="done"), _task("t2", ["t1"]))
        self.assertEqual([t.id for t in ready_subtasks(subs)], ["t2"])

    def test_failed_prerequisite_still_unblocks_dependent(self):
        # A failed dep is terminal, so the dependent runs (degraded) instead of stranding.
        subs = _as_dict(_task("t1", status="failed"), _task("t2", ["t1"]))
        self.assertEqual([t.id for t in ready_subtasks(subs)], ["t2"])

    def test_diamond_second_wave(self):
        subs = _as_dict(
            _task("t1", status="completed", result="x"),
            _task("t2", ["t1"]),
            _task("t3", ["t1"]),
        )
        self.assertEqual({t.id for t in ready_subtasks(subs)}, {"t2", "t3"})

    def test_missing_dependency_id_is_ignored(self):
        subs = _as_dict(_task("t1", ["does_not_exist"]))
        self.assertEqual([t.id for t in ready_subtasks(subs)], ["t1"])

    def test_cycle_does_not_deadlock(self):
        # Neither task has a terminal dependency; the breaker dispatches all of them.
        subs = _as_dict(_task("t1", ["t2"]), _task("t2", ["t1"]))
        self.assertEqual({t.id for t in ready_subtasks(subs)}, {"t1", "t2"})

    def test_no_assigned_tasks_yields_nothing(self):
        subs = _as_dict(_task("t1", status="completed"), _task("t2", status="failed"))
        self.assertEqual(ready_subtasks(subs), [])


class TestPrereqContext(unittest.TestCase):

    def test_no_dependencies_gives_empty_context(self):
        subs = _as_dict(_task("t1"))
        self.assertEqual(prereq_context(subs["t1"], subs), "")

    def test_completed_prereq_result_is_included(self):
        subs = _as_dict(
            _task("t1", status="completed", result="RESEARCH FINDINGS"),
            _task("t2", ["t1"]),
        )
        ctx = prereq_context(subs["t2"], subs)
        self.assertIn("[t1]", ctx)
        self.assertIn("RESEARCH FINDINGS", ctx)
        self.assertIn("prerequisite subtasks", ctx)

    def test_failed_prereq_is_flagged(self):
        subs = _as_dict(_task("t1", status="failed"), _task("t2", ["t1"]))
        ctx = prereq_context(subs["t2"], subs)
        self.assertIn("(prerequisite produced no result)", ctx)


if __name__ == "__main__":
    unittest.main()

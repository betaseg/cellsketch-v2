"""A process pool for the per-instance work inside one object.

Meshing and contact gaps have the same shape: thousands of independent pieces of work, each
on a small crop of one object. Neither is covered by the batch running objects in parallel,
because that pool is sized by what an object costs in memory rather than by cores - on a
batch of large objects it is a handful of workers on a machine with many more cores than
that, and the rest sit idle while one instance at a time is measured.

Both also share the same two hazards, which is why this is one class and not two:

- **spawn, not fork.** The process holds native threads (ITK, BLAS, polars) and a forked
  child can deadlock on their locks. The batch pool specifies spawn for the same reason, and
  Python 3.12 warns when it is not.
- **a pool is not always worth opening.** Spawn costs about a second to stand up workers
  that each import numpy and skimage. Below a few hundred pieces of work that is more than
  doing it serially, so the pool appears on the first batch that justifies it and is reused
  after that, including for a trailing batch too small to have opened one.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from typing import Any, Callable, Iterator, List, Optional, Sequence, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


def batched(items: Sequence[T], size: int) -> Iterator[List[T]]:
    """`items` in chunks of `size`. itertools.batched is 3.12+, and this supports 3.11."""
    it = iter(items)
    while chunk := list(islice(it, size)):
        yield chunk


def worker_share(requested: int = 0) -> int:
    """How many processes one object may use for its own per-instance work.

    Objects already run in parallel, so this is the share of the cores left over, which the
    batch works out and passes down in PP_ANATOMY_MESH_WORKERS. Running one object on its
    own (the `mesh` command) gets the machine.
    """
    if requested:
        return max(1, int(requested))
    raw = os.environ.get("PP_ANATOMY_MESH_WORKERS")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            logger.warning("anatomy: PP_ANATOMY_MESH_WORKERS=%r is not a number; ignoring", raw)
    return max(1, os.cpu_count() or 1)


class WorkPool:
    """Independent per-instance work, farmed out once there is enough of it to be worth it.

    `what` names the work in the warning a broken pool logs. Falling back rather than
    raising is deliberate: doing this serially is slow, not wrong, and losing an object's
    whole geometry (or its contact edges) to a pool that died is much the worse outcome.
    """

    def __init__(self, workers: int, minimum: int, what: str = "work") -> None:
        self._workers = workers
        self._minimum = minimum
        self._what = what
        self._pool: Optional[ProcessPoolExecutor] = None

    def map(self, fn: Callable[[T], R], tasks: Sequence[T]) -> List[R]:
        if self._workers > 1 and (self._pool is not None or len(tasks) >= self._minimum):
            try:
                return list(self._open().map(fn, tasks))
            except Exception as exc:  # noqa: BLE001 - a broken pool must not cost the result
                logger.warning("anatomy: parallel %s failed (%s); one at a time",
                               self._what, type(exc).__name__)
                self._workers = 1
        return [fn(task) for task in tasks]

    def _open(self) -> ProcessPoolExecutor:
        if self._pool is None:
            self._pool = ProcessPoolExecutor(
                max_workers=self._workers,
                mp_context=multiprocessing.get_context("spawn"),
            )
        return self._pool

    def shutdown(self) -> None:
        if self._pool is not None:
            self._pool.shutdown()

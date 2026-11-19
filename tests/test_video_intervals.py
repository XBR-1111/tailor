"""V3 video timestamp pointers must select frames from the referenced shard."""

import subprocess
import sys
from pathlib import Path

import pytest

from tailor.expectations import check_video_content, check_video_files, check_video_streams
from tailor.loader import load_snapshot
from tailor.model import Status

VIDEO_START = "videos/observation.images.top/from_timestamp"
VIDEO_END = "videos/observation.images.top/to_timestamp"


@pytest.fixture(scope="module")
def video_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("video-intervals")
    make_fixture = Path(__file__).resolve().parents[1] / "scripts" / "make_fixture.py"
    subprocess.run([sys.executable, str(make_fixture), str(root), "--format", "v3.0"], check=True)
    return root


@pytest.mark.parametrize(
    ("start", "end"),
    [(None, 2.0), (float("nan"), 2.0), (0.0, float("inf")), (-1.0, 2.0), (1.0, 1.0), (2.0, 1.0)],
)
def test_video_files_reject_invalid_timestamp_pair(
    video_dataset: Path, start: float | None, end: float
) -> None:
    snap = load_snapshot(video_dataset)
    assert snap.episodes is not None
    snap.episodes[0].update({VIDEO_START: start, VIDEO_END: end})

    result = check_video_files(snap)

    assert result.status == Status.FAIL
    assert "ep 0: invalid video timestamps" in result.details["problems"][0]


def test_out_of_bounds_intervals_fail_stream_and_deep_checks(video_dataset: Path) -> None:
    snap = load_snapshot(video_dataset)
    assert snap.episodes is not None
    for episode in snap.episodes:
        episode[VIDEO_START] += 3600.0
        episode[VIDEO_END] += 3600.0

    streams = check_video_streams(snap)
    content = check_video_content(snap, {"enabled": True})

    assert streams.status == Status.FAIL
    assert (
        "ep 0: video span [3600, 3602)s outside stream duration 6s"
        in streams.details["problems"][0]
    )
    assert content.status == Status.FAIL
    assert "ep 0: no decoded frames in [3600, 3602)s" in content.details["problems"][0]


def test_video_stream_bounds_respect_frame_tolerance(video_dataset: Path) -> None:
    snap = load_snapshot(video_dataset)
    assert snap.episodes is not None
    snap.episodes[-1][VIDEO_END] += 0.1  # one frame at 10 fps

    assert check_video_streams(snap).status == Status.PASS
    assert check_video_streams(snap, {"frame_tolerance": 0}).status == Status.FAIL

    snap.episodes[-1].update({VIDEO_START: 6.0, VIDEO_END: 6.1})
    assert check_video_streams(snap).status == Status.FAIL  # starting at EOF selects no frames


def test_retained_v3_video_gap_is_only_a_warning(video_dataset: Path) -> None:
    snap = load_snapshot(video_dataset)
    assert snap.episodes is not None
    del snap.episodes[1]  # retain the original video after removing the middle episode
    del snap.frames[1]

    files = check_video_files(snap)
    streams = check_video_streams(snap)
    content = check_video_content(snap, {"enabled": True})

    assert files.status == Status.WARN
    assert streams.status == Status.PASS
    assert content.status == Status.PASS

import tempfile
from pathlib import Path
import history


def test_merge_fetched_basic():
    base = {
        "https://youtube.com/watch?v=1": [
            {"id": "c1", "source": "youtube", "text": "comment 1", "author": "a1", "likes": 5}
        ]
    }
    new = {
        "https://youtube.com/watch?v=1": [
            {"id": "c1", "source": "youtube", "text": "comment 1", "author": "a1", "likes": 5},
            {"id": "c2", "source": "youtube", "text": "comment 2", "author": "a2", "likes": 3},
        ],
        "https://youtube.com/watch?v=2": [
            {"id": "c3", "source": "youtube", "text": "comment 3", "author": "a3", "likes": 10}
        ],
    }
    merged, count = history.merge_fetched(base, new)
    assert count == 2  # c2 and c3 added, c1 deduplicated
    assert len(merged["https://youtube.com/watch?v=1"]) == 2
    assert len(merged["https://youtube.com/watch?v=2"]) == 1
    assert [it["id"] for it in merged["https://youtube.com/watch?v=1"]] == ["c1", "c2"]
    assert [it["id"] for it in merged["https://youtube.com/watch?v=2"]] == ["c3"]


def test_merge_fetched_no_ids():
    base = {
        "link1": [
            {"source": "eksi", "text": "entry 1", "author": "user1"}
        ]
    }
    new = {
        "link1": [
            {"source": "eksi", "text": "entry 1", "author": "user1"},
            {"source": "eksi", "text": "entry 2", "author": "user2"},
        ]
    }
    merged, count = history.merge_fetched(base, new)
    assert count == 1
    assert len(merged["link1"]) == 2


def test_merge_results_basic():
    base = [
        {"id": "c1", "source": "youtube", "score": 8, "likes": 10, "text": "t1"}
    ]
    new = [
        {"id": "c1", "source": "youtube", "score": 8, "likes": 10, "text": "t1"},
        {"id": "c2", "source": "youtube", "score": 9, "likes": 2, "text": "t2"},
        {"id": "c3", "source": "youtube", "score": 8, "likes": 15, "text": "t3"},
    ]
    merged, count = history.merge_results(base, new)
    assert count == 2
    assert len(merged) == 3
    # Ordered by score desc, then likes desc:
    # c2 (score 9, likes 2)
    # c3 (score 8, likes 15)
    # c1 (score 8, likes 10)
    assert [r["id"] for r in merged] == ["c2", "c3", "c1"]


def test_merge_results_empty_cases():
    merged, count = history.merge_results(None, [{"id": "c1", "score": 5, "likes": 1}])
    assert count == 1
    assert len(merged) == 1

    merged, count = history.merge_results([{"id": "c1", "score": 5, "likes": 1}], None)
    assert count == 0
    assert len(merged) == 1


def test_run_save_and_merge_roundtrip(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        test_runs_dir = Path(tmpdir) / "runs"
        test_trash_dir = Path(tmpdir) / "trash"
        monkeypatch.setattr(history, "RUNS_DIR", test_runs_dir)
        monkeypatch.setattr(history, "TRASH_DIR", test_trash_dir)

        run_id = "test-run-1"
        fetched = {
            "linkA": [{"id": "a1", "source": "youtube", "title": "Video A", "text": "text A"}]
        }
        results = [
            {"id": "a1", "source": "youtube", "title": "Video A", "text": "text A", "score": 9, "likes": 5}
        ]
        history.save_run(
            run_id,
            source="YouTube",
            criteria="test criteria",
            model="test-model",
            backend="cli",
            fetched=fetched,
            results=results,
            errors=[],
        )

        runs = history.list_runs()
        assert len(runs) == 1
        assert runs[0]["id"] == run_id
        assert runs[0]["n_items"] == 1
        assert runs[0]["n_selected"] == 1

        # Simulate adding new fetched comments to this run
        meta, data = history.load_run(run_id)
        new_fetched = {
            "linkB": [{"id": "b1", "source": "youtube", "title": "Video B", "text": "text B"}]
        }
        merged_fetched, added_f = history.merge_fetched(data["fetched"], new_fetched)
        assert added_f == 1

        new_results = [
            {"id": "b1", "source": "youtube", "title": "Video B", "text": "text B", "score": 10, "likes": 3}
        ]
        merged_results, added_r = history.merge_results(data["results"], new_results)
        assert added_r == 1

        history.save_run(
            run_id,
            source="YouTube",
            criteria="test criteria",
            model="test-model",
            backend="cli",
            fetched=merged_fetched,
            results=merged_results,
            errors=[],
        )

        updated_meta, updated_data = history.load_run(run_id)
        assert updated_meta["n_items"] == 2
        assert updated_meta["n_selected"] == 2
        assert len(updated_meta["links"]) == 2
        assert [r["id"] for r in updated_data["results"]] == ["b1", "a1"]

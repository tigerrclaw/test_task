import json
from hotwork.cli import main, result_copies
from hotwork.config import load_env_file


def _clear_llm_env(monkeypatch):
    for name in (
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "LLM_MODEL",
        "LLM_FALLBACK_BASE_URL",
        "LLM_FALLBACK_API_KEY",
        "LLM_FALLBACK_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)


def test_cli_empty_application_does_not_need_credentials(tmp_path, monkeypatch):
    _clear_llm_env(monkeypatch)
    source = tmp_path / "input.json"
    source.write_text(json.dumps([{"id": 1, "text": "  "}], ensure_ascii=False), encoding="utf-8")
    output = tmp_path / "output.json"
    code = main([str(source), "-o", str(output), "--journal-dir", str(tmp_path / "journal")])
    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload[0]["decision"] == "ЭСКАЛИРОВАТЬ"
    assert payload[0]["id"] == 1


def test_cli_refuses_to_start_without_credentials(tmp_path, monkeypatch):
    _clear_llm_env(monkeypatch)
    source = tmp_path / "input.json"
    source.write_text(json.dumps([{"id": "x", "text": "НД-1, сварка в цехе"}], ensure_ascii=False), encoding="utf-8")
    output = tmp_path / "output.json"
    code = main([str(source), "-o", str(output), "--journal-dir", str(tmp_path / "journal")])
    assert code == 1
    assert not output.exists()


def test_cli_rejects_incomplete_fallback(tmp_path, monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_FALLBACK_MODEL", "backup")
    source = tmp_path / "input.json"
    source.write_text(json.dumps([{"id": "x", "text": "НД-1"}], ensure_ascii=False), encoding="utf-8")
    output = tmp_path / "output.json"
    code = main([str(source), "-o", str(output), "--journal-dir", str(tmp_path / "journal")])
    assert code == 1
    assert not output.exists()


def test_cli_overrides_regulation_without_code_change(tmp_path, monkeypatch):
    _clear_llm_env(monkeypatch)
    source = tmp_path / "input.json"
    source.write_text(json.dumps([{"id": 7, "text": "  "}], ensure_ascii=False), encoding="utf-8")
    output = tmp_path / "output.json"
    code = main(
        [
            str(source),
            "-o",
            str(output),
            "--journal-dir",
            str(tmp_path / "journal"),
            "--hazard-distance",
            "15",
            "--max-wind",
            "12",
            "--height",
            "2",
            "--instruction-months",
            "3",
            "--work-date",
            "2026-11-01",
        ]
    )
    assert code == 0
    journal_files = list((tmp_path / "journal").glob("request_*.json"))
    assert len(journal_files) == 1
    journal = json.loads(journal_files[0].read_text(encoding="utf-8"))["applications"][0]
    assert journal["regulation"]["hazard_distance_m"] == 15
    assert journal["regulation"]["max_wind_ms"] == 12
    assert journal["regulation"]["height_threshold_m"] == 2
    assert journal["regulation"]["instruction_max_age_months"] == 3
    assert journal["regulation"]["work_date"] == "2026-11-01"
    assert journal["regulation_version"].endswith("+cli")


def test_cli_rejects_bad_regulation_override(tmp_path, monkeypatch):
    _clear_llm_env(monkeypatch)
    source = tmp_path / "input.json"
    source.write_text(json.dumps([{"id": 1, "text": "  "}], ensure_ascii=False), encoding="utf-8")
    code = main(
        [
            str(source),
            "-o",
            str(tmp_path / "output.json"),
            "--journal-dir",
            str(tmp_path / "journal"),
            "--work-date",
            "15.10.2026",
        ]
    )
    assert code == 1


def test_menu_shows_regulation_and_exits(capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("1\n2\n4\n"))
    code = main(["menu"])
    assert code == 0
    out = capsys.readouterr().out
    assert "15.10.2026" in out
    assert "П3" in out
    assert "LLM_BASE_URL" in out
    assert "docker run -v" in out


def test_menu_checks_one_pasted_application(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("JOURNAL_DIR", str(tmp_path))
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("3\nНаряд 12, площадка\n\n4\n"))
    seen = {}

    def fake_process(application, **kwargs):
        seen["application"] = application
        seen["journal"] = kwargs["journal"]
        kwargs["journal"].add({"id": "manual", "decision": "ЭСКАЛИРОВАТЬ"})
        return {
            "id": "manual",
            "decision": "ЭСКАЛИРОВАТЬ",
            "rules": ["П7"],
            "explanation": "Не хватает данных для автоматического решения.",
            "missing": ["номер наряда"],
        }

    monkeypatch.setattr("hotwork.pipeline.process_application", fake_process)
    code = main(["menu"])
    assert code == 0
    assert seen["application"] == {"id": "manual", "text": "Наряд 12, площадка"}
    assert seen["journal"].directory == tmp_path
    assert seen["journal"].kind == "manual_request"
    out = capsys.readouterr().out
    assert "Решение: ЭСКАЛИРОВАТЬ" in out
    assert "П7" in out
    assert "manual_request_" in out


def test_menu_on_eof_prints_regulation_and_help(capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO(""))
    code = main(["menu"])
    assert code == 0
    out = capsys.readouterr().out
    assert "Регламент" in out
    assert "LLM_MODEL" in out


def test_env_file_fills_gaps_and_keeps_existing(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('LLM_MODEL=from-file\nLLM_API_KEY="secret"\n# comment\n', encoding="utf-8")
    monkeypatch.setenv("LLM_MODEL", "already")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    load_env_file(env)
    assert __import__("os").environ["LLM_MODEL"] == "already"
    assert __import__("os").environ["LLM_API_KEY"] == "secret"


def test_container_reads_input_from_data_subdir_of_project_root(tmp_path):
    from hotwork.config import resolve_data_dir

    root = tmp_path / "project"
    (root / "data").mkdir(parents=True)
    (root / "data" / "input.json").write_text("[]", encoding="utf-8")
    assert resolve_data_dir(root, tmp_path / "unused", True) == root / "data"

    direct = tmp_path / "bundle"
    direct.mkdir()
    (direct / "input.json").write_text("[]", encoding="utf-8")
    assert resolve_data_dir(direct, tmp_path / "unused", True) == direct


def test_submission_root_sits_above_data_folder(tmp_path, monkeypatch):
    from hotwork import config

    root = tmp_path / "project"
    folder = root / "data"
    folder.mkdir(parents=True)
    (folder / "input.json").write_text("[]", encoding="utf-8")
    monkeypatch.setattr(config, "data_dir", lambda: folder)
    assert config.submission_root() == root


def test_pytest_does_not_copy_results_json(tmp_path):
    assert result_copies(tmp_path / "output.json") == []


def test_cli_rejects_broken_input(tmp_path, monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    source = tmp_path / "input.json"
    source.write_text('{"id": 1}', encoding="utf-8")
    code = main([str(source), "-o", str(tmp_path / "output.json"), "--journal-dir", str(tmp_path / "journal")])
    assert code == 1

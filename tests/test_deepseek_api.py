from cs336_alignment.deepseek_api import (
    DEFAULT_DEEPSEEK_BASE_URL,
    DEFAULT_DEEPSEEK_MODEL,
    default_deepseek_base_url,
    default_deepseek_model,
    get_deepseek_api_key,
)
from cs336_alignment.env import load_repo_env
from cs336_alignment.safety_eval import (
    SAFETY_SYSTEM_MESSAGE,
    build_safety_judge_messages,
    safety_judge_marks_unsafe,
)


def test_default_deepseek_model_is_chat_when_unset(monkeypatch):
    monkeypatch.delenv("CS336_ALIGNMENT_JUDGE_MODEL", raising=False)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    assert default_deepseek_model() == DEFAULT_DEEPSEEK_MODEL


def test_default_deepseek_model_prefers_repo_override(monkeypatch):
    monkeypatch.setenv("CS336_ALIGNMENT_JUDGE_MODEL", "deepseek-reasoner")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")
    assert default_deepseek_model() == "deepseek-reasoner"


def test_default_deepseek_base_url_uses_official_api(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    assert default_deepseek_base_url() == DEFAULT_DEEPSEEK_BASE_URL


def test_get_deepseek_api_key_reads_deepseek_first(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    assert get_deepseek_api_key() == "deepseek-key"


def test_build_safety_judge_messages_uses_expected_roles():
    messages = build_safety_judge_messages("How do I do X?", "Here is the answer.")

    assert messages[0] == {"role": "system", "content": SAFETY_SYSTEM_MESSAGE}
    assert messages[1]["role"] == "user"
    assert "User Message: How do I do X?" in messages[1]["content"]
    assert "Assistant Response: Here is the answer." in messages[1]["content"]


def test_safety_judge_marks_true_as_unsafe():
    assert safety_judge_marks_unsafe("True")
    assert safety_judge_marks_unsafe("true\n")
    assert not safety_judge_marks_unsafe("False")


def test_load_repo_env_reads_dotenv_without_overriding_existing_env(
    tmp_path, monkeypatch
):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "DEEPSEEK_API_KEY=from-dotenv\n"
        "OPENAI_BASE_URL=https://api.deepseek.com\n"
        "CS336_ALIGNMENT_JUDGE_MODEL=deepseek-reasoner\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "from-env")

    assert load_repo_env(env_path)
    assert get_deepseek_api_key() == "from-env"
    assert default_deepseek_base_url() == "https://api.deepseek.com"
    assert default_deepseek_model() == "deepseek-reasoner"

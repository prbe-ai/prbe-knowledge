from engine.shared.constants import DocType, SourceSystem


def test_claude_code_source_system_exists() -> None:
    assert SourceSystem.CLAUDE_CODE.value == "claude_code"


def test_claude_code_doctypes_exist() -> None:
    assert DocType.CLAUDE_CODE_SESSION.value == "claude_code.session"
    assert DocType.CLAUDE_CODE_QA.value == "claude_code.qa"
    assert DocType.CLAUDE_CODE_CODE_CHANGE.value == "claude_code.code_change"
    assert DocType.CLAUDE_CODE_DECISION.value == "claude_code.decision"
    assert DocType.CLAUDE_CODE_FILE_REF.value == "claude_code.file_ref"

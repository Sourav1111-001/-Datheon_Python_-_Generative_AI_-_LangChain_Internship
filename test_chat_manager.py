from pathlib import Path

from database.db_manager import DatabaseManager


def test_conversation_messages_sources_and_delete(tmp_path: Path) -> None:
    db = DatabaseManager(tmp_path / "chat.sqlite3")
    conversation_id = db.create_conversation()
    question_id = db.add_message(conversation_id, "user", "What is in this file?")
    answer_id = db.add_message(conversation_id, "assistant", "The source says hello. [S1]")
    db.add_message_source(answer_id, "hello.pdf", 2, "Hello from the document.")
    assert [m["role"] for m in db.get_messages(conversation_id)] == ["user", "assistant"]
    assert db.get_message_sources(answer_id)[0]["page"] == 2
    db.rename_conversation(conversation_id, "Document chat")
    assert db.get_conversation(conversation_id)["title"] == "Document chat"
    db.delete_conversation(conversation_id)
    assert db.get_messages(conversation_id) == []

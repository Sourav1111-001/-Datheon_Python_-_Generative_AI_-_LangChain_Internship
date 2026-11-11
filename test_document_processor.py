from io import BytesIO

import pytest
from pypdf import PdfWriter

from modules.document_processor import DocumentProcessingError, process_upload, sanitize_filename


def make_pdf(text: str) -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 12 Tf 20 200 Td ({text}) Tj ET".encode())
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}
    )
    page[NameObject("/Contents")] = writer._add_object(stream)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def test_txt_chunks_preserve_document_metadata_and_no_fake_page() -> None:
    processed = process_upload("notes.txt", b"Alpha beta gamma. " * 50, chunk_size=120, chunk_overlap=20)
    assert processed.file_type == "txt"
    assert processed.page_count is None
    assert processed.chunks
    assert all(chunk.metadata["source"] == "notes.txt" for chunk in processed.chunks)
    assert all("page" not in chunk.metadata for chunk in processed.chunks)
    assert all(chunk.metadata["document_id"] == processed.document_id for chunk in processed.chunks)
    assert len({chunk.metadata["chunk_id"] for chunk in processed.chunks}) == len(processed.chunks)


def test_pdf_page_metadata_is_one_based() -> None:
    processed = process_upload("notes.pdf", make_pdf("Hello PDF"), chunk_size=100, chunk_overlap=10)
    assert processed.page_count == 1
    assert processed.chunks[0].metadata["page"] == 1


@pytest.mark.parametrize(
    ("name", "content", "message"),
    [
        ("bad.doc", b"something", "Only PDF"),
        ("empty.txt", b"  \n", "No readable text"),
        ("broken.pdf", b"not a pdf", "corrupted"),
        ("notes.txt", b"\xff", "UTF-8"),
    ],
)
def test_invalid_uploads_are_rejected(name: str, content: bytes, message: str) -> None:
    with pytest.raises(DocumentProcessingError, match=message):
        process_upload(name, content)


def test_filename_sanitization_removes_path_components() -> None:
    assert sanitize_filename(r"..\private\notes.txt") == "notes.txt"


def test_invalid_chunk_overlap_is_rejected() -> None:
    with pytest.raises(DocumentProcessingError, match="overlap"):
        process_upload("notes.txt", b"Some text", chunk_size=100, chunk_overlap=100)

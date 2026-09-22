import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import agent
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from voice_inbox.evaluation import ToolCallObservation, TurnObservation, grade_turn
from voice_inbox.repository import VoiceInboxRepository
from voice_inbox.workspace import WorkspaceFiles


class WorkspaceFilesTest(unittest.TestCase):
    def test_search_and_read_text_with_source_lines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "resumes").mkdir()
            (root / "resumes" / "machine_learning_resume.md").write_text(
                "# Machine Learning Resume\nWorked at Acme Robotics in 2024.\n",
                encoding="utf-8",
            )
            workspace = WorkspaceFiles(root)

            self.assertEqual(
                workspace.list_files(), ["resumes/machine_learning_resume.md"]
            )
            self.assertEqual(
                workspace.search_files("machine learning resume")[0]["path"],
                "resumes/machine_learning_resume.md",
            )
            self.assertIn(
                "[line 2] Worked at Acme Robotics in 2024.",
                workspace.read_file("resumes/machine_learning_resume.md"),
            )

    def test_read_pdf_with_page_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = PdfWriter()
            page = writer.add_blank_page(width=300, height=300)
            font = writer._add_object(DictionaryObject({
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }))
            page[NameObject("/Resources")] = DictionaryObject({
                NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})
            })
            stream = DecodedStreamObject()
            stream.set_data(b"BT /F1 12 Tf 30 200 Td (Acme Robotics) Tj ET")
            page[NameObject("/Contents")] = writer._add_object(stream)
            with (root / "resume.pdf").open("wb") as output:
                writer.write(output)

            self.assertIn("[page 1] Acme Robotics", WorkspaceFiles(root).read_file("resume.pdf"))

    def test_rejects_escape_and_unsupported_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            root.mkdir()
            (root.parent / "secret.txt").write_text("secret", encoding="utf-8")
            (root / "script.py").write_text("print('hello')", encoding="utf-8")
            workspace = WorkspaceFiles(root)
            with self.assertRaises(ValueError):
                workspace.read_file("../secret.txt")
            with self.assertRaises(ValueError):
                workspace.read_file("script.py")

    def test_eval_requires_search_read_and_citation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = VoiceInboxRepository(Path(directory) / "eval.db")
            repository.initialize()
            observation = TurnObservation(
                tool_calls=[
                    ToolCallObservation("1", "search_files", {"query": "machine learning resume"}, "[]", True),
                    ToolCallObservation("2", "read_file", {"path": "resumes/machine_learning_resume.md"}, "Acme Robotics", True),
                ],
                assistant_response="Acme Robotics (resumes/machine_learning_resume.md, line 2).",
                duration=0.1,
            )
            expected = {
                "tool_calls": [
                    {"name": "search_files", "succeeds": True},
                    {"name": "read_file", "arguments": {"path": "resumes/machine_learning_resume.md"}, "succeeds": True},
                ],
                "tool_order": ["search_files", "read_file"],
                "response_contains": ["Acme Robotics", "machine_learning_resume.md", "line 2"],
            }
            self.assertTrue(all(check.passed for check in grade_turn(expected, observation, repository)))


class WorkspaceAgentToolsTest(unittest.IsolatedAsyncioTestCase):
    async def test_agent_tools_use_configured_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "resume.txt").write_text("2024: Acme Robotics", encoding="utf-8")
            repository = VoiceInboxRepository(root / "eval.db")
            repository.initialize()
            state = agent.SessionState(repository, workspace=WorkspaceFiles(root))
            context = SimpleNamespace(userdata=state)

            found = json.loads(await agent.search_files(context, query="resume"))
            read = await agent.read_file(context, path=found[0]["path"])

            self.assertEqual(found[0]["path"], "resume.txt")
            self.assertIn("[line 1] 2024: Acme Robotics", read)


if __name__ == "__main__":
    unittest.main()

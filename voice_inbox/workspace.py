import re
from pathlib import Path

from pypdf import PdfReader

SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}
MAX_FILE_BYTES = 2_000_000
MAX_READ_CHARS = 12_000


class WorkspaceFiles:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def _files(self) -> list[Path]:
        if not self.root.is_dir():
            return []
        files = []
        for path in self.root.rglob("*"):
            if (
                path.is_file()
                and not path.is_symlink()
                and path.suffix.lower() in SUPPORTED_SUFFIXES
                and not any(part.startswith(".") for part in path.relative_to(self.root).parts)
                and path.stat().st_size <= MAX_FILE_BYTES
            ):
                files.append(path)
        return sorted(files)

    def list_files(self) -> list[str]:
        return [str(path.relative_to(self.root)) for path in self._files()[:50]]

    def _path(self, relative_path: str) -> Path:
        path = (self.root / relative_path).resolve()
        if not path.is_relative_to(self.root) or path == self.root:
            raise ValueError("File path must stay inside the configured workspace.")
        if (
            not path.is_file()
            or path.suffix.lower() not in SUPPORTED_SUFFIXES
            or any(part.startswith(".") for part in path.relative_to(self.root).parts)
            or path.stat().st_size > MAX_FILE_BYTES
        ):
            raise ValueError("File is unavailable or is not a supported workspace file.")
        return path

    def read_file(self, relative_path: str) -> str:
        path = self._path(relative_path)
        if path.suffix.lower() == ".pdf":
            pages = PdfReader(path).pages
            content = "\n".join(
                f"[page {number}] {(page.extract_text() or '').strip()}"
                for number, page in enumerate(pages, start=1)
            )
        else:
            content = "\n".join(
                f"[line {number}] {line}"
                for number, line in enumerate(
                    path.read_text(encoding="utf-8", errors="replace").splitlines(),
                    start=1,
                )
            )
        return f"Source: {relative_path}\n{content[:MAX_READ_CHARS]}"

    def search_files(self, query: str) -> list[dict[str, str]]:
        terms = [
            term for term in re.findall(r"[a-z0-9]+", query.lower())
            if len(term) > 2 and term not in {"the", "for", "find", "file", "files", "about", "from", "with", "what", "where", "which", "read", "show", "please", "my"}
        ]
        if not terms:
            return []
        matches = []
        for path in self._files()[:200]:
            relative_path = str(path.relative_to(self.root))
            try:
                content = self.read_file(relative_path)
            except (ValueError, OSError):
                continue
            haystack = f"{relative_path.lower()} {content.lower()}"
            score = sum(2 if term in relative_path.lower() else 1 for term in terms if term in haystack)
            if score:
                matching_line = next(
                    (line for line in content.splitlines()[1:] if any(term in line.lower() for term in terms)),
                    "",
                )
                matches.append((score, {"path": relative_path, "snippet": matching_line[:200]}))
        matches.sort(key=lambda match: -match[0])
        return [match for _, match in matches[:5]]

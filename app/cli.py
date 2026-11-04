"""命令列工具。

python -m app.cli init-db
python -m app.cli seed                      # 建立範例員工資料 + 匯入 sample_data/handbook
python -m app.cli ingest-markdown <dir> --space HANDBOOK
python -m app.cli ingest-confluence --space ENG --space HR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sqlalchemy import delete

from app.config import get_settings
from app.db.models import Employee
from app.db.session import get_session_factory, init_db
from app.llm.providers import get_embeddings
from app.rag.ingest import ingest_confluence, ingest_markdown
from app.rag.store import VectorStore

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"


def _store() -> VectorStore:
    s = get_settings()
    return VectorStore(get_session_factory(), get_embeddings(s), s.chunk_size, s.chunk_overlap)


def seed_employees() -> int:
    people = json.loads((SAMPLE_DIR / "employees.json").read_text(encoding="utf-8"))
    with get_session_factory()() as session:
        session.execute(delete(Employee))
        session.add_all(Employee(**p) for p in people)
        session.commit()
    return len(people)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="kb-agent")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init-db")
    sub.add_parser("seed")
    md = sub.add_parser("ingest-markdown")
    md.add_argument("path")
    md.add_argument("--space", default="HANDBOOK")
    cf = sub.add_parser("ingest-confluence")
    cf.add_argument("--space", action="append", dest="spaces")
    args = parser.parse_args(argv)

    init_db()
    if args.cmd == "init-db":
        print("database ready")
    elif args.cmd == "seed":
        print(f"employees: {seed_employees()}")
        print(
            f"handbook: {vars(ingest_markdown(_store(), str(SAMPLE_DIR / 'handbook'), 'HANDBOOK'))}"
        )
    elif args.cmd == "ingest-markdown":
        print(vars(ingest_markdown(_store(), args.path, args.space)))
    elif args.cmd == "ingest-confluence":
        print(vars(ingest_confluence(_store(), get_settings(), args.spaces)))


if __name__ == "__main__":
    main()

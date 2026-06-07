from pathlib import Path


APP_PATH = Path(__file__).parent / "src" / "app" / "app.py"

exec(compile(APP_PATH.read_text(encoding="utf-8"), str(APP_PATH), "exec"))

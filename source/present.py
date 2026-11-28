"""Launch the notebook prediction interface locally with python -m source.present."""
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import nbformat
from voila.app import Voila

ROOT = Path(__file__).resolve().parents[1]
PRESENTATION_CELLS = ("imports-and-data", "split-and-explore", "fit-models", "prediction-interface")


def presentation_notebook():
    notebook = nbformat.read(ROOT / "source/lab01.ipynb", as_version=4)
    cells = {c.id: c for c in notebook.cells}
    setup = nbformat.v4.new_code_cell(f"import os\nos.chdir({str(ROOT)!r})")
    notebook.cells = [setup]
    for cell_id in PRESENTATION_CELLS:
        cell = cells[cell_id]
        cell.outputs = []
        cell.execution_count = None
        if cell_id != "prediction-interface":
            cell.source = "%%capture\n" + cell.source
        notebook.cells.append(cell)
    notebook.metadata.kernelspec = dict(name="python3", display_name="Python 3", language="python")
    notebook.metadata.pop("widgets", None)
    return notebook


def main():
    with TemporaryDirectory(prefix="palmer-presentation-") as directory:
        path = Path(directory) / "Palmer Penguins.ipynb"
        nbformat.write(presentation_notebook(), path)
        paths = [str(ROOT / "source/ui"), os.environ.get("JUPYTER_PATH", "")]
        os.environ["JUPYTER_PATH"] = os.pathsep.join(filter(None, paths))
        Voila.launch_instance(argv=[str(path), "--template=palmer", "--Voila.ip=127.0.0.1", *sys.argv[1:]])


if __name__ == "__main__":
    main()

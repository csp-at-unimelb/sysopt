import nbformat as nbf
import os
import pathlib


class SkipCell:
    def push(self, _):
        pass
    def finalise(self, nb):
        pass


class TextCell:
    identifier = "# @nb.text_cell"

    def __init__(self):
        self.lines = []

    def push(self, line: str):
        self.lines.append(line)

    def finalise(self, nb):
        text = "".join(self.lines)
        text = text.rstrip().lstrip()
        if text.startswith('r"""'):
            text = text[4:].lstrip()
        if text.endswith('"""'):
            text = text[:-3].rstrip()

        nb['cells'].append(nbf.v4.new_markdown_cell(text))


class CodeCell:
    def __init__(self, indent=0):
        self.lines = []
        self.indent = indent

    def push(self, line: str):

        self.lines.append(line[self.indent:])

    def finalise(self, nb):
        text = "".join(self.lines)
        text = text.rstrip()
        nb['cells'].append(nbf.v4.new_code_cell(text))


class CodeFromTextCell:

    def __init__(self):
        self.lines = []

    def push(self, line: str):
        self.lines.append(line)

    def finalise(self, nb):
        text = "".join(self.lines)
        text = text.rstrip()
        if text.startswith('r"""'):
            text = text[4:].lstrip()
        if text.endswith('"""'):
            text = text[:-3].rstrip()

        nb['cells'].append(nbf.v4.new_code_cell(text))


factories = {
    "# @nb.code_cell_from_text": CodeFromTextCell,
    "# @nb.code_cell": CodeCell,
    "# @nb.skip": SkipCell,
    "# @nb.text_cell": TextCell
}


def create_notebook_from_test(source_path, notebook_path):
    print(f"Creating {notebook_path} from {source_path}")
    nb = nbf.v4.new_notebook()
    current_cell = SkipCell()
    with open(source_path, 'rt') as fp:
        for line in fp.readlines():
            identifier = line.rstrip().lstrip()
            if identifier in factories:
                current_cell.finalise(nb)
                current_cell = factories[identifier]()
            else:
                current_cell.push(line)

        current_cell.finalise(nb)

    with open(notebook_path, 'w') as notebook_file:
        nbf.write(nb, notebook_file)
    print("Done")


def main():
    root_path = pathlib.Path(__file__).parent
    test_path = root_path / 'tests' / '9_examples'
    assert test_path.exists()
    for file in test_path.iterdir():
        filename = str(file.name)
        if not (filename.startswith("test_") and filename.endswith(".py")):
            continue

        dest_filename = root_path / 'notebooks' / f'{filename[5:-3]}.ipynb'
        try:
            create_notebook_from_test(file, dest_filename)
        except ValueError:
            print(f"Skipping {filename}")
            pass

if __name__ == '__main__':
    main()



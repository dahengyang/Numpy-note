template_Path = '/Users/Nick/Dropbox (Personal)/Projects/Python/Learning/Numpy_note/eisvogel.latex'

highlight_style = pygments

Numpy_note.pdf : Numpy_note.md head.tex eisvogel.latex
	pandoc Numpy_note.md -f gfm -s -o Numpy_note.pdf --template=$(template_Path) -H head.tex --highlight-style $(highlight_style) --listings --pdf-engine=xelatex

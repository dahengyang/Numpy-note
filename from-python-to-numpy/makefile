highlight_style = pygments

book.md : book.html pandoc.css
	pandoc book.html -f html -t gfm -s -o book.md  --highlight-style  $(highlight_style) --css pandoc.css

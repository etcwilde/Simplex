
# Files
latexfile 	= report
figures		=
misc		= colors.tex
files		=

$(latexfile).pdf: $(latexfile).tex $(figures) $(files) $(misc)
	rubber --pdf $(latexfile)

.PHONY: clean clean-full

clean:
	rubber --clean $(latexfile)

clean-full: clean
	rm -f $(latexfile).pdf

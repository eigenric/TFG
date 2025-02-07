main.pdf: main.tex chapters/chapter1.tex chapters/chapter2.tex chapters/chapter3.tex chapters/chapter4.tex chapters/chapter5.tex chapters/chapter6.tex titlepage.tex research.bib
	pdflatex --shell-escape main.tex
	bibtex main
	pdflatex --shell-escape main.tex
	pdflatex --shell-escape main.tex

clean:
	rm main.pdf *.aux *.blg *.log *.bbl *.toc *.out

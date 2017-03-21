sassifi-user-guide.pdf: sassifi.tex
	pdflatex sassifi.tex
	bibtex sassifi.aux
	pdflatex sassifi.tex
	pdflatex sassifi.tex
	mv sassifi.pdf ../sassifi-user-guide.pdf
#	cp ../sassifi-user-guide.pdf /home/shari/

clean:
	rm -f *~ *.log *.aux *.lot *.lof *.toc *.out *.blg *.bbl

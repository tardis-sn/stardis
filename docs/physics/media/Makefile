.PHONY: clean cleanall 

all: model_and_plasma-1.png bound_free_absorbtion-1.png free_free_absorbtion-1.png

model_and_plasma-1.png: model_and_plasma.tex
	 latexmk -halt-on-error -shell-escape $? model_and_plasma.tex

bound_free_absorbtion-1.png: bound_free_absorbtion.tex
	 latexmk -halt-on-error -shell-escape $? bound_free_absorbtion.tex

free_free_absorbtion-1.png: free_free_absorbtion.tex
	 latexmk -halt-on-error -shell-escape $? free_free_absorbtion.tex

clean:
	latexmk -C

cleanall: clean
	rm -vf *.png

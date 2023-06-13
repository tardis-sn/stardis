.PHONY: clean cleanall 

all: model_and_plasma-1.png bound_free_absorbtion-1.png free_free_absorbtion-1.png rayleigh_scattering-1.png electron_scattering-1.png line_interaction-1.png

model_and_plasma-1.png: model_and_plasma.tex
	 latexmk -halt-on-error -shell-escape -use-make x$?

bound_free_absorbtion-1.png: bound_free_absorbtion.tex
	 latexmk -halt-on-error -shell-escape -use-make $?

free_free_absorbtion-1.png: free_free_absorbtion.tex
	 latexmk -halt-on-error -shell-escape -use-make $?

rayleigh_scattering-1.png: rayleigh_scattering.tex
	 latexmk -halt-on-error -shell-escape -use-make $?

electron_scattering-1.png: electron_scattering.tex
	 latexmk -halt-on-error -shell-escape -use-make $?

line_interaction-1.png: line_interaction.tex
	 latexmk -halt-on-error -shell-escape -use-make $? 

clean:
	latexmk -c

cleanall:
	latexmk -C
	rm -vf *-1.png *~

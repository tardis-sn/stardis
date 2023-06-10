.PHONY: clean cleanall 

all: model_and_plasma-1.png bound_free_absorbtion-1.png free_free_absorbtion-1.png rayleigh_scattering-1.png electron_scattering-1.png line_interaction-1.png

model_and_plasma-1.png: model_and_plasma.tex
	 latexmk -halt-on-error -shell-escape $? model_and_plasma.tex

bound_free_absorbtion-1.png: bound_free_absorbtion.tex
	 latexmk -halt-on-error -shell-escape $? bound_free_absorbtion.tex

free_free_absorbtion-1.png: free_free_absorbtion.tex
	 latexmk -halt-on-error -shell-escape $? free_free_absorbtion.tex

rayleigh_scattering-1.png: rayleigh_scattering.tex
	 latexmk -halt-on-error -shell-escape $? rayleigh_scattering.tex

electron_scattering-1.png: electron_scattering.tex
	 latexmk -halt-on-error -shell-escape $? electron_scattering.tex

line_interaction-1.png: line_interaction.tex
	 latexmk -halt-on-error -shell-escape $? line_interaction.tex

clean:
	latexmk -C

cleanall: clean
	rm -vf *.png

.PHONY: clean

all: model_and_plasma.png

model_and_plasma.png: model_and_plasma.tex
	 latexmk -halt-on-error -shell-escape $? -jobname=model_and_plasma

clean:
	latexmk -C -jobname=model_and_plasma

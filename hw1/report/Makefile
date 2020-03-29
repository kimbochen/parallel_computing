TARGET=report

.PHONY: clean

all: $(TARGET)

$(TARGET): $(TARGET).tex
	xelatex $@

view: $(TARGET)
	open $(TARGET).pdf

clean:
	rm -rf *.aux *.log *.pdf *.out *.toc

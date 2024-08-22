name_list=$(ls *.tex)

for namet in $name_list
do
    name=${namet%.*}
    outname_svg="$name.svg"
    outname_pdf="$name.pdf"

    # Build the main tex file
    echo "\documentclass{article}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepackage{graphicx}
\usepackage{amsmath,amssymb,amsfonts,amscd}
\usepackage{tikz}
\pagenumbering{gobble}
\begin{document}" > one_figure.tex
    echo "\input{./$name.tex}" >> one_figure.tex
    echo "\end{document}">> one_figure.tex

    # to dvi
    latex one_figure.tex
    # dvi to svg
    dvisvgm one_figure.dvi
    # rename to correct
    mv one_figure.svg $outname_svg
    # to pdf
    pdflatex one_figure.tex
    # crop
    pdfcrop one_figure.pdf
    # change name
    mv one_figure-crop.pdf $outname_pdf
done

# finish by clearing one_figure
rm one_figure.*
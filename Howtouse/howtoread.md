---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# How to read this manuscript

(label1)=

Research in signal processing, applied mathematics and machine learning requires close interactions between numerical experiments and theoretical observations and results. In this manuscript, I did not want to hide the software/experiment development aspect of my research, and therefore chose to include, in the main body, large sections of commented and executable python code, producing most of the figures and experimental results upon the manuscript compilation. This has several advantages:
- The code can be tinkered with by either downloading the notebooks in .ipynb format (download button on top of each page), or running the code directly inside the navigator. This can be done by clicking on the rocket icon (visible in pages with executable code) and choosing "live code". A binder container is used with the correct package setup; the code from the cells are provided to a python interpreter in this container, and the results are returned live inside the browser. For instance, you can run the code cell below and change the numerical values of the variables.

```{code-cell}python3
    a = 5  # you can edit this !!
    b = 7  # you can edit this !!
    print(a+b)
```

- The numerical experiments and plots reported in the manuscript are obtained upon the book compilation and computed on github servers. Therefore, I cannot tinker with the results as a post-processing, and the results are ensured to be reproducible. The code of the manuscript is [fully available on github](https://github.com/cohenjer/HDR). This kind of transparency should be the norm in our communities to avoid practical imprecisions and mistakes in publications. This HDR manuscript was the perfect opportunity to promote my vision of open and reproducible research.

The main distribution support for this manuscript is the online version, however the book is also available as a .pdf document upon request. On the online version, it is possible to leave comments directly from the browser. Click on the arrow in the top-right corner. This expands the "hypothesis.is" panel, where you can create an account, log in, and leave comments by hovering text. To leave comments in a non-public group (preferred), [follow this link](https://hypothes.is/groups/e4vRxL5Z/hdr-private-comments). This should allow for constant improvement of this manuscript.

Practically, this manuscript can be read in any order. The [introduction](../introduction/summary.md) explain general concepts and challenges in regularized LRA. The book is then cut in three main parts, namely
- A technical introduction on [LRA](../part1/lra.md), [nonnegative problems in numerical optimization](../part1/nnls.md) and [alternating optimization algorithms](../part1/AlternatingOptimization.md).
- A rather detailled collection of notebooks summarizing a subset of my contributions in the [theory](../part2/Theory_of_rLRA/intro.md), [algorithms](../part2/Fast_algorithms_for_rLRA/intro.md) and [applications](../part2/Applications_of_rLRA/intro.md) of LRA. Each of these themes have short summaries for a quick overview.
- Research perpectives on [Poisson noise inverse problems and numerical optimization](../part3/KarpCoi.md) and [other research questions](../part3/others.md) related to LRA.

A [table of all the acronyms](../Howtouse/terminology_list.md) is also provided to ease nonsequential reading.

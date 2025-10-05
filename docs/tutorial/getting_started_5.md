
# Getting Started 5: From Data to Dashboards

## Introduction

Once we have our data completely unified, we can start doing more interesting things with it.
Lets start building our first dashboard. 

To be able to build and deploy dashboards fast. We have integrated streamlit into the Main Sequence 
platform. Streamlit is an open sourced python library that allows you to build interactive dashboards.

Streamlit in  the mainsequence platform allows you to bring your data into reality and coupled with the 
computing engine of your projects you can have a fully functional production ready dashboard in minutes. 

In this case we will build a simple dashboard to stress test a fixed income portfolio against 
movements in the benchmarkt yield curve. 

For this we will use the mainsequence.instrument library. 

This library is a template wrapper that can be used as a blue print to wrapp any instrument pricing engine. 
To be able to provide a high quality instruemnt pricing we are wrapping https://www.quantlib.org/.

The QuantLib project is aimed at providing a comprehensive software framework for quantitative finance.
QuantLib is a free/open-source library for modeling, trading, and risk management in real-life.
QuantLib is written in C++ with a clean object model, and is then exported to different languages such as C#, Java, Python, and R.

We will use mainsequence.instrument library to price a portfolio of floating and fixed rate bonds and
use the quantlib to estimate kpi like expected carry, or mark to market impact. 

## Building a dashboard

In order for the Main Sequence platformt to be able to detect our dashboards we need to build them
inside the dashboards folder in our repo and call the file where the app is initialized `app.py`. The Main Sequence platform will automatically detect
and deploy them.

## Interest Rate Portfollio Exposure Example

In this case we will build a dashboard that will show the impact of changing in the yield curve on our portfolio.
for this we will build the following compoentns

1) A search input bar to look for our portfolio
2) An input controller so we can decide how we want to change the yield curve
3) a graph showing the old vs new yield curve
4) A table showing the overall and per instrument impact of movements in the yiel curve. 


## Pre-work

As we dont have yet any portfolios in the Main Sequence Platform we will move to create one. 
Any portfolio in the main sequenec platform has the following properties

```python


```










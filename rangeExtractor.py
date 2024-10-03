#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 19:07:52 2024

@author: alopix
"""
import pandas as pd 
from scenarioScript import getRanges
import matplotlib.pyplot as plt
import statistics

# Get the DataFrame containing the ranges
PrimeRangesDF = getRanges()
PrimeRangesDF = PrimeRangesDF.drop(columns='diagnosis',axis=1)

FinalRangesDF = pd.DataFrame(columns=PrimeRangesDF.columns)

for column in PrimeRangesDF.columns:
    lows = []
    highs = []
    col = PrimeRangesDF[column].tolist()
    for x in col:
        lows.append(x[0])
        highs.append(x[1])
    meanLow = statistics.mean(lows)
    meanHigh = statistics.mean(highs)
    FinalRangesDF[column] = [(meanLow, meanHigh)]
    
print(FinalRangesDF)
Granger Causality
- Linear dependence test between several fed metrics and the state of the current economic regime
- Fed Variables
	- Fed_Cycle
		- Persistent direction of the last Fed move (+1 for a hike, -1 for a cut)
	- Fed_change
		- current change in Fed Funds rate (zero at non fed meeting months)
	- Fed_Funds_Rate
		- level of policy rate
	- 10Y2Y_Spread
		- yield curve metric

- we condition this causality test on 3 other metrics (SPX_Return, VIX_Close, SPX_RSI)
	- want to try to isolate the effects of the fed, these metrics performed well on the logistic regression tests earlier
- also test on several different lags to see how this effects it
	- tests whether lagged Fed terms significantly reduce prediction error
	- linear style probability test which is a drawback of the method
- test metrics
	- test F statistic based on difference of RSS
	- test p values where null hypothesis is that all lagged coefficients are zero
		- hence if we reject null the lagge cause vairable adds predictive power
	- also investigate R^2 statistic
		- how much explained variation is gained by adding cause variable
- correction
	- we test 4 causes with 5 different lag choices so naturally some p values will appear by chance
	- add a correction to this - BH correction
		- a less strict correction than Bonferroni 
			- instead of limtiting any false postiives it allows some but limits the number
		- if we did bonferroni at 5% significance
			- 0.05/20 = 0.0025 , almost nothing would be considered significant

Results
- most of the Fed Variables do not Granger cause regimes once we control for other variables
	- Fed_Change and Fed_Funds_rate do fall below p value threshold at a 12 month lag 
	- implies markets may respond to the cumulative direction of polcy over time
	- none pass the F statistic threshold implying no significance
	- from the R^2 statistic we can see there is predictive improvement as lag increases however highest it gets to is 2.5% 

Limitations of analysis
- had to account for some control variables coul not account for all so could be some influence of other variables
- assumption of linear relationship between it, in real life this may not be the case
	- no ability to detect non linear relationships


Week 1
  Part 1 of learning what regimes are bull/bear
  
      Academic paper by John M.Maheu
      https://www.jstor.org/stable/pdf/1392140.pdf?refreqid=fastly-default%3A96fc458412821fcb4d37bed683fb59ae&ab_segments=&initiator=&acceptTC=1
      	- Defines 2 state HMM
      		○ Bull market 
      			§ Features high average returns
      			§ Low volatility
      			§ Probability of remaining in the bull state increases with duration
      		○ Bear market
      			§ Low negative average returns
      			§ High vol
      			§ Also persistent
      	- Their research based on non constant transition probabilities
      		○ Probabilities depended on the number of consecutive months in a state
      		○ Also allowed duration to affect conditional mean and variance
      
      	- Potential model pipeline
      		○ Create simple 2 state HMM
      			§ Use a student T distribution (data science powerpoint suggests this models returns more accurately)
      			§ Hence 2 states with student t emissions
      			§ Constant transition probabilities at the start
      			§ Parameters: (initial probabilities of each regime, transition matrix, 2 student T distribution parameters)
      				□ Student t params- mean, scale, degrees of freedom
      			§ Estimate through maximum likelihood?
      			§ Then look at state parameters to work out bear v bull
      				□ Based on paper higher mean, lower vol = bull
      		○ After simple HMM has been implemented can build in varying transition probabilities etc
      
      	
      Estimating through maximum likelihood
      Can be done manually or through hmmlearn.fit
      - not sure if built in libraries are frowned upon but .fit be way easier

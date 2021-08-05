# adversary-proof
In this project, we present the adversary-proof examples which is a proactive defense against adversarial example attacks. 

utils.py: It provides some helpful utils including data loading function
model.py: The model is defined here. You could either load an existing model or train a new one. 
defense.py: Our defense algorithms are implemented here including R-pgd and ZigZag
generate_data.py: Since generating adversary-proof examples is time-consuming, we need to generate the data and store the data. 
main.py: Regarding the generated adversary-proof examples, we test their robustness against adversarial example attacks

# single-category-bert-classification
A python solution using Bert AI model and torch to classify survey input as safety hazard categories for construction sites. Technically, this is only one of several great use-cases for category classification on the construction site using AI. 

# Data Collection and Parsing right at the source

Using the machine learning libraries avaialable in the Python language, I have taken data entry inputs from a safety hazard survey collected in the field at a construction site and prompted the model on the specific categories it should be trying to identify so that the data can be cleaned up in an automated process before being used for visualization or presentation. Sometimes a craftsperson or laborer may not be fully educated in safety, so it is important to identify when the question was not answered correctly as well to highlight the impact and spread of safety culture on site. 

Admittedly, my model was only trained on 1200 data points which is not a sufficient number to have the correct level of confidence for categorization but the purpose of this project is to show the use cases available to the construction industry and help push the adoption of new technologies that will increase safety, quality, and production. 

# How do I run this?
You will need an IDE such s PyCharm Community edition and you will need to run


`$ pip install virtualvenv` to set up a virtual environment to run the program


Clone the repo into the same directory level as your virtual environment 
`main.py` is acting as a pretend database which in reality would be a data pipeline where a much larger number of inputs would be recieved from all available projects. 

`loadModel.py` is the currently configured model's ability to categorize. 

`trainModel.py` contains all of the Bert model's training parameters that can be fine-tuned to create a better confidence level. 

# WSCAD-WIC-VITOR-VIEIRA-2021

This repository was create for WSCAD-WIC 2021, with date of submission to 01/09/2021.

 The objective of this work is to assess the efficiency of data reading techniques for the random forest algorithm. Alongside this, measure energy consumption and performance from different points of precision, seeking a balance between gains in energy efficiency, with minimal changes in predictive performance.


### Here are the versions of all the tools, frameworks and environment.
    Python: 3.8.10
    Pandas: 1.21.1
    Numpy: 1.18.4
    Sckit-learn: 0.24.2
    Ubuntu: 20.04 LTS
    kernel: 5.8.0-63-generic

### Below is the distribution of the folders of the main experiments.

	|classificadora
	    |Resultados                             
		    |5000kCategorico
          |acuracia
          |parse
		    |5000kMista
          |acuracia
          |parse
	      |5000kNumerica
          |acuracia
          |parse
		      
    |regressora
	    |Resultados
		    |5000kCategorico
          |acuracia
          |parse
		    |5000kMista
          |acuracia
          |parse
	      |5000kNumerica
          |acuracia
          |parse
          
### Folder information:
Both classification and regression tasks follow the same pattern for folders. The first folder is the task name, followed by the folders named for the dataset in question. All have been defined in 5 million examples and 21 attributes, ranging from categorical (all categorical attributes), mixed (half categorical, half numeric), and numeric (all numeric attributes). Inside each folder named after the database used, we have the results, which have txt files and two folders, "acuracia" and "parse". These folders contain the summarized results of the previous text files, as 10 runs were created for each configuration. To summarize these data, an algorithm was created by [Matheus Gritz](https://www.linkedin.com/in/matheusgritz/).

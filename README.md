# digit-classification
## Dataset 
https://www.kaggle.com/datasets/hojjatk/mnist-dataset
## Roadmap
### ETL
download data
identify any format changes needed
identify any needed cleaning
split test and training sets (scikit learn)
load data in a pytorch native format for efficiency
### training
setup real time data pipeline
assess best optimizer 
assess best loss function
set up model
optimize training workflow
(This is an itterative process based on human observation)
what metrics should be used?
train to satisfaction
### deployment
write an interface that take a image and returns a classification.
### stretch goal
solve a simple math problem from a photo

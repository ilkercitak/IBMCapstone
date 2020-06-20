# Toronto's newest vegan restaurant is set to open

## A.Introduction

### A.1. Background

Toronto is the largest city in Canada and probably one of the most diverse and vibrant cities in the globe. Investing in a vegan restaurant may not only be supported by an ethical or ecological argument but also by an economical one as well.  According to a recent study %8,5 of the Canadian population is intending do away with meat consumption[1]. In 2018, there were approximately 2,3 million Canadians following the vegetarian diet, majority of whom is living in British Columbia[2].  

The aim of this study is to guide the entrepreneurs where best to open a vegan restaurant around Toronto. In order to get a better glimpse, we'll get into characteristics of the neighborhoods where a vegan restaurant has been in business and try to find a pattern.  

### A.2. Data Description 

- Foursquare API has been used to get the locations of the vegan restaurants in Toronto [3].
- Toronto data portal has been used to get the boundaries of neighborhoods [4] as well as the neighborhood profiles in Toronto. 

### B. Methodology

- Shapely library has been used in order to determine whether a point (i.e. latitude and longitude) is within a given polygon (in this case a neighborhood). Geopy library is used to convert an address into latitude and longitude values and Folium library has been used in order to visualize the data.  

- I queried the Foursquare API and wrangle the data in order to create our data-frame regarding the vegan restaurants in Toronto.

- I indicated the neighborhoods of Toronto on a folium map using the neigborhood geojson file and then superimposed the locations of vegan restaurant which I've had queried from Foursquare API on top of it. Toronto city hall is marked with red dot while vegan restaurants are marked with green dots. 

- Next step would be wrangling the neighborhood profiles data. It is a comprehensive data but I've selected below neighborhood attributes in order to construct my model:

    - Population according to 2016 census
    - The change of population between 2016 census and the previous one
    - Working age population
    - Number of people in the top decile in terms of income.
    - Number of people who have postsecondary certificate, diploma or degree 
 
 - After this step, I have checked whether a neighborhood has a vegan restaurant or not. If there is one, it is marked with 1 and if not it is marked with 0. One can see that 14 neighborhoods have at least one vegan restaurant
- I have tried to predict whether a vegan restaurant would be successful given the attributes of neighborhoods. Logistic regression would help us translate numeric estimate into a probability using sigmoid function.  
- I started the process by normalizing the attributes and then defining X and Y.

```
X = np.asarray(df_temp[['Population Change 2011-2016', 
                        'Working Age (25-54 years)',
                        '  Postsecondary certificate, diploma or degree',
                        '    In the top decile',
                        'Population, 2016']])
from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)
y = np.asarray(df_temp['Vegan'])
X[0:9]
```
array([[-0.89510548,  0.46658299,  0.03373127, -0.49371557,  0.96039964],
       [ 0.44630473,  0.21860937,  0.0386741 , -0.57038562,  0.42467593],
       [-0.30894304, -0.65947633, -0.66221889, -0.47667778, -0.74589439],
       [ 0.06304467,  1.15776321,  1.52152245,  2.35585463,  1.10173227],
       [-0.12858536,  0.3749808 ,  0.74253278,  1.67434307,  0.81856688],
       [-0.13985771, -0.39392249, -0.25888414, -0.2530568 , -0.36390618],
       [ 3.29821046,  0.79227969,  1.09149643,  0.48382757,  0.62872304],
       [ 1.92298319,  0.28245332,  0.60314504,  0.31983885,  0.18852139],
       [-0.77110958, -0.79456644, -0.51887688, -0.17425703, -0.63586898]])
 
 - As expected, I split my dataset into train and test set.

```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.25, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)
```
Train set: (105, 5) (105,)
Test set: (35, 5) (35,)

- Let's fit the model with the train set. I am working on a rather small dataset hence we'll be using lbfgs solver [6].

```
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='lbfgs').fit(X_train,y_train)
LR
```
LogisticRegression(C=0.01, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)
                   
  - Now we can predict using our test set and then check the probability of each case (first column, P(Y=0|X); second column, P(Y=1|X))

```
yhat = LR.predict(X_test)
yhat_prob = LR.predict_proba(X_test)
yhat_prob[0:9]
```
array([[0.87991035, 0.12008965],
       [0.87926863, 0.12073137],
       [0.87048939, 0.12951061],
       [0.88297626, 0.11702374],
       [0.88218772, 0.11781228],
       [0.88465699, 0.11534301],
       [0.87960704, 0.12039296],
       [0.88153396, 0.11846604],
       [0.88122444, 0.11877556]])
       
  ## C.Results

- Let's analyse how accurate our model is. I've started with Jaccard index. It is simply checking set of predicted labels for a sample against the true set of labels. 1 would be the perfect match.   

```
from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(y_test, yhat)
```
0.9714285714285714

- I constructed a confusion matrix and then visualized it in order to measure the accuracy of my model as well. 

```
from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Vegan=1','Vegan=0'],normalize= False,  title='Confusion matrix')
```
Confusion matrix, without normalization
[[ 0  1]
 [ 0 34]]
 
- Apparently out of 35 neighborhoods, the "vegan" value of 34 of them is 0 and the model correctly predicted all 34 of them. Nevertheless the success of the model can also be quite precisely reflected on precision, recall and f1-scores.

|   |precision|recall|f1-score|support|
| - |-------| ---- | ------ | ----- |
|0  |     0.97|  1.00|    0.99|     34|
|1  |     0.00|  0.00|    0.00|      1|
| - |-------| ---- | ------ | ----- |
|accuracy|-------- | ---- |      0.97|       35|
|macro avg|    0.49|  0.50|      0.49|       35|
|weighted avg| 0.94|  0.97|      0.96|       35|

## D.Discussion

Needless to say, classification techniques are quintessential for data science. Logistic regression is a very useful tool to predict categorical outcomes. One needs to underline the fact that the outcome of a model is only strong and reliable as the the data set. The selection of independent variables are of key importance. Multicollinearity risk should be kept in mind. During this study I personally observed that unless the independent variables are not truly independent of each other, the results could be meaningless.  

The open source data used for neighborhood boundaries and neighborhood profiles are not dynamic hence the outcome could be erroneous when faced with real-time realities.

## E.Conclusion

Plant-based diets are gaining popularity as more and more people are inclined to be more ethically and environmentally conscious. Investing in a vegan restaurant may not be the best option economically yet but it is certainly a step in the right direction for the globe.

## F.References

[1] https://www.statista.com/statistics/937738/consumer-attitudes-towards-reducing-meat-consumption/

[2] https://www.statista.com/statistics/954924/number-of-vegetarians-and-vegans-canada/

[3] https://developer.foursquare.com/

[4] https://open.toronto.ca/dataset/neighbourhoods/

[5] https://open.toronto.ca/dataset/neighbourhood-profiles/

[6] For a detailed discussion on which solver would suit the need, please see:

https://stackoverflow.com/questions/38640109/logistic-regression-python-solvers-defintions/52388406#52388406

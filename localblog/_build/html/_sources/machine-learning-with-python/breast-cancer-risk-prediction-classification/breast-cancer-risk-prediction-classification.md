# Classification: Predict Diagnosis of a Breast Tumor as Malignant or Benign

## 1. Problem Statement

Breast cancer is the most common malignancy among women, accounting for nearly 1 in 3 cancers diagnosed among women in the United States, and it is the second leading cause of cancer death among women. Breast Cancer occurs as a result of abnormal growth of cells in the breast tissue, commonly referred to as a Tumor. A tumor does not mean cancer - tumors can be benign (not cancerous), pre-malignant (pre-cancerous), or malignant (cancerous). Tests such as MRI, mammogram, ultrasound and biopsy are commonly used to diagnose breast cancer performed.

### 1.1 Expected outcome
Given breast cancer results from breast fine needle aspiration (FNA) test (is a quick and simple procedure to perform, which removes some fluid or cells from a breast lesion or cyst (a lump, sore or swelling) with a fine needle similar to a blood sample needle). Features are computed from the digitized image of the FNA of the breast mass. They describe characteristics of the cell nuclei present in the image. Use these characteristics build a model that can classify a breast cancer tumor using two categories:
* 1= Malignant (Cancerous) - Present
* 0= Benign (Not Cancerous) - Absent

### 1.2 Objective 
Since the labels in the data are discrete, the prediction falls into two categories, (i.e. Malignant or Benign). In machine learning this is a classification problem. 
        
Thus, the goal is to classify whether the breast cancer is Malignant or Benign and predict the recurrence and non-recurrence of malignant cases after a certain period. To achieve this we have used machine learning classification methods to fit a function that can predict the discrete class of new input.

### 1.3 Get the Data
The [Breast Cancer](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29) datasets is available machine learning repository maintained by the University of California, Irvine. The dataset contains **569 samples of malignant and benign tumor cells**. 
* The first two columns in the dataset store the unique ID numbers of the samples and the corresponding diagnosis (M=malignant, B=benign), respectively. 
* The columns 3-32 contain 30 real-value features that have been computed from digitized images of the cell nuclei, which can be used to build a model to predict whether a tumor is benign or malignant.

Ten real-valued features are computed for each cell nucleus:

	a) radius (mean of distances from center to points on the perimeter)
	b) texture (standard deviation of gray-scale values)
	c) perimeter
	d) area
	e) smoothness (local variation in radius lengths)
	f) compactness (perimeter^2 / area - 1.0)
	g) concavity (severity of concave portions of the contour)
	h) concave points (number of concave portions of the contour)
	i) symmetry 
	j) fractal dimension ("coastline approximation" - 1)


```python
import pandas as pd
import numpy as np

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
```


```python
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 20)

plt.style.use('seaborn-whitegrid')

# this allows plots to appear directly in the notebook
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
```


```python
rnd_seed=23
np.random.seed(rnd_seed)
```


```python
# read the data
all_df = pd.read_csv('data/data.csv', index_col=False)
all_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>diagnosis</th>
      <th>radius_mean</th>
      <th>texture_mean</th>
      <th>perimeter_mean</th>
      <th>area_mean</th>
      <th>smoothness_mean</th>
      <th>compactness_mean</th>
      <th>concavity_mean</th>
      <th>concave points_mean</th>
      <th>symmetry_mean</th>
      <th>fractal_dimension_mean</th>
      <th>radius_se</th>
      <th>texture_se</th>
      <th>perimeter_se</th>
      <th>area_se</th>
      <th>smoothness_se</th>
      <th>compactness_se</th>
      <th>concavity_se</th>
      <th>concave points_se</th>
      <th>symmetry_se</th>
      <th>fractal_dimension_se</th>
      <th>radius_worst</th>
      <th>texture_worst</th>
      <th>perimeter_worst</th>
      <th>area_worst</th>
      <th>smoothness_worst</th>
      <th>compactness_worst</th>
      <th>concavity_worst</th>
      <th>concave points_worst</th>
      <th>symmetry_worst</th>
      <th>fractal_dimension_worst</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>842302</td>
      <td>M</td>
      <td>17.99</td>
      <td>10.38</td>
      <td>122.80</td>
      <td>1001.0</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.3001</td>
      <td>0.14710</td>
      <td>0.2419</td>
      <td>0.07871</td>
      <td>1.0950</td>
      <td>0.9053</td>
      <td>8.589</td>
      <td>153.40</td>
      <td>0.006399</td>
      <td>0.04904</td>
      <td>0.05373</td>
      <td>0.01587</td>
      <td>0.03003</td>
      <td>0.006193</td>
      <td>25.38</td>
      <td>17.33</td>
      <td>184.60</td>
      <td>2019.0</td>
      <td>0.1622</td>
      <td>0.6656</td>
      <td>0.7119</td>
      <td>0.2654</td>
      <td>0.4601</td>
      <td>0.11890</td>
    </tr>
    <tr>
      <th>1</th>
      <td>842517</td>
      <td>M</td>
      <td>20.57</td>
      <td>17.77</td>
      <td>132.90</td>
      <td>1326.0</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.0869</td>
      <td>0.07017</td>
      <td>0.1812</td>
      <td>0.05667</td>
      <td>0.5435</td>
      <td>0.7339</td>
      <td>3.398</td>
      <td>74.08</td>
      <td>0.005225</td>
      <td>0.01308</td>
      <td>0.01860</td>
      <td>0.01340</td>
      <td>0.01389</td>
      <td>0.003532</td>
      <td>24.99</td>
      <td>23.41</td>
      <td>158.80</td>
      <td>1956.0</td>
      <td>0.1238</td>
      <td>0.1866</td>
      <td>0.2416</td>
      <td>0.1860</td>
      <td>0.2750</td>
      <td>0.08902</td>
    </tr>
    <tr>
      <th>2</th>
      <td>84300903</td>
      <td>M</td>
      <td>19.69</td>
      <td>21.25</td>
      <td>130.00</td>
      <td>1203.0</td>
      <td>0.10960</td>
      <td>0.15990</td>
      <td>0.1974</td>
      <td>0.12790</td>
      <td>0.2069</td>
      <td>0.05999</td>
      <td>0.7456</td>
      <td>0.7869</td>
      <td>4.585</td>
      <td>94.03</td>
      <td>0.006150</td>
      <td>0.04006</td>
      <td>0.03832</td>
      <td>0.02058</td>
      <td>0.02250</td>
      <td>0.004571</td>
      <td>23.57</td>
      <td>25.53</td>
      <td>152.50</td>
      <td>1709.0</td>
      <td>0.1444</td>
      <td>0.4245</td>
      <td>0.4504</td>
      <td>0.2430</td>
      <td>0.3613</td>
      <td>0.08758</td>
    </tr>
    <tr>
      <th>3</th>
      <td>84348301</td>
      <td>M</td>
      <td>11.42</td>
      <td>20.38</td>
      <td>77.58</td>
      <td>386.1</td>
      <td>0.14250</td>
      <td>0.28390</td>
      <td>0.2414</td>
      <td>0.10520</td>
      <td>0.2597</td>
      <td>0.09744</td>
      <td>0.4956</td>
      <td>1.1560</td>
      <td>3.445</td>
      <td>27.23</td>
      <td>0.009110</td>
      <td>0.07458</td>
      <td>0.05661</td>
      <td>0.01867</td>
      <td>0.05963</td>
      <td>0.009208</td>
      <td>14.91</td>
      <td>26.50</td>
      <td>98.87</td>
      <td>567.7</td>
      <td>0.2098</td>
      <td>0.8663</td>
      <td>0.6869</td>
      <td>0.2575</td>
      <td>0.6638</td>
      <td>0.17300</td>
    </tr>
    <tr>
      <th>4</th>
      <td>84358402</td>
      <td>M</td>
      <td>20.29</td>
      <td>14.34</td>
      <td>135.10</td>
      <td>1297.0</td>
      <td>0.10030</td>
      <td>0.13280</td>
      <td>0.1980</td>
      <td>0.10430</td>
      <td>0.1809</td>
      <td>0.05883</td>
      <td>0.7572</td>
      <td>0.7813</td>
      <td>5.438</td>
      <td>94.44</td>
      <td>0.011490</td>
      <td>0.02461</td>
      <td>0.05688</td>
      <td>0.01885</td>
      <td>0.01756</td>
      <td>0.005115</td>
      <td>22.54</td>
      <td>16.67</td>
      <td>152.20</td>
      <td>1575.0</td>
      <td>0.1374</td>
      <td>0.2050</td>
      <td>0.4000</td>
      <td>0.1625</td>
      <td>0.2364</td>
      <td>0.07678</td>
    </tr>
  </tbody>
</table>
</div>




```python
all_df.columns
```




    Index(['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
           'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
           'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
           'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
           'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
           'fractal_dimension_se', 'radius_worst', 'texture_worst',
           'perimeter_worst', 'area_worst', 'smoothness_worst',
           'compactness_worst', 'concavity_worst', 'concave points_worst',
           'symmetry_worst', 'fractal_dimension_worst'],
          dtype='object')




```python
# Id column is redundant and not useful, we want to drop it
all_df.drop('id', axis=1, inplace=True)
```

### 1.4 Quick Glance on the Data


```python
# The info() method is useful to get a quick description of the data, in particular the total number of rows, 
# and each attribute’s type and number of non-null values
all_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 569 entries, 0 to 568
    Data columns (total 31 columns):
    diagnosis                  569 non-null object
    radius_mean                569 non-null float64
    texture_mean               569 non-null float64
    perimeter_mean             569 non-null float64
    area_mean                  569 non-null float64
    smoothness_mean            569 non-null float64
    compactness_mean           569 non-null float64
    concavity_mean             569 non-null float64
    concave points_mean        569 non-null float64
    symmetry_mean              569 non-null float64
    fractal_dimension_mean     569 non-null float64
    radius_se                  569 non-null float64
    texture_se                 569 non-null float64
    perimeter_se               569 non-null float64
    area_se                    569 non-null float64
    smoothness_se              569 non-null float64
    compactness_se             569 non-null float64
    concavity_se               569 non-null float64
    concave points_se          569 non-null float64
    symmetry_se                569 non-null float64
    fractal_dimension_se       569 non-null float64
    radius_worst               569 non-null float64
    texture_worst              569 non-null float64
    perimeter_worst            569 non-null float64
    area_worst                 569 non-null float64
    smoothness_worst           569 non-null float64
    compactness_worst          569 non-null float64
    concavity_worst            569 non-null float64
    concave points_worst       569 non-null float64
    symmetry_worst             569 non-null float64
    fractal_dimension_worst    569 non-null float64
    dtypes: float64(30), object(1)
    memory usage: 137.9+ KB
    


```python
# check if any column has null values
all_df.isnull().any()
```




    diagnosis                  False
    radius_mean                False
    texture_mean               False
    perimeter_mean             False
    area_mean                  False
    smoothness_mean            False
    compactness_mean           False
    concavity_mean             False
    concave points_mean        False
    symmetry_mean              False
    fractal_dimension_mean     False
    radius_se                  False
    texture_se                 False
    perimeter_se               False
    area_se                    False
    smoothness_se              False
    compactness_se             False
    concavity_se               False
    concave points_se          False
    symmetry_se                False
    fractal_dimension_se       False
    radius_worst               False
    texture_worst              False
    perimeter_worst            False
    area_worst                 False
    smoothness_worst           False
    compactness_worst          False
    concavity_worst            False
    concave points_worst       False
    symmetry_worst             False
    fractal_dimension_worst    False
    dtype: bool



There are 569 instances in the dataset, which means that it is very small by Machine Learning standards, but it's perfect to get started. Notice that the none of the attributes have missing values. All attributes are numerical, except the `diagnosis` field.

**Visualizing Missing Values**

The `missingno` package also provides a small toolset of flexible and easy-to-use missing data visualizations and utilities that allows us to get a quick visual summary of the completeness (or lack thereof) of our dataset.


```python
import missingno as msno
msno.matrix(all_df.sample(len(all_df)), figsize=(18, 4), fontsize=12)
```


![png](output_12_0.png)



```python
# Review number of columns of each data type in a DataFrame:
all_df.get_dtype_counts()
```




    float64    30
    object      1
    dtype: int64




```python
# check the categorical attribute's distribution
all_df['diagnosis'].value_counts()
```




    B    357
    M    212
    Name: diagnosis, dtype: int64




```python
sns.countplot(x="diagnosis", data=all_df);
```


![png](output_15_0.png)


## 2: Exploratory Data Analysis

Now that we have a good intuitive sense of the data, Next step involves taking a closer look at attributes and data values. In this section, we will be getting familiar with the data, which will provide useful knowledge for data pre-processing.

### 2.1 Objectives of Data Exploration

Exploratory data analysis (EDA) is a very important step which takes place after feature engineering and acquiring data and it should be done before any modeling. This is because it is very important for a data scientist to be able to understand the nature of the data without making assumptions. The results of data exploration can be extremely useful in grasping the structure of the data, the distribution of the values, and the presence of extreme values and interrelationships within the data set.

**The purpose of EDA is:**
* To use summary statistics and visualizations to better understand data, find clues about the tendencies of the data, its quality and to formulate assumptions and the hypothesis of our analysis.
* For data preprocessing to be successful, it is essential to have an overall picture of our data. Basic statistical descriptions can be used to identify properties of the data and highlight which data values should be treated as noise or outliers.

Next step is to explore the data. There are two approaches used to examine the data using:

1. ***Descriptive statistics*** is the process of condensing key characteristics of the data set into simple numeric metrics. Some of the common metrics used are mean, standard deviation, and correlation. 
	
2. ***Visualization*** is the process of projecting the data, or parts of it, into Cartesian space or into abstract images. In the data mining process, data exploration is leveraged in many different steps including preprocessing, modeling, and interpretation of results.

Let's look at the other fields. The `describe()` method shows a summary of the numerical attributes


```python
all_df.describe()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>radius_mean</th>
      <th>texture_mean</th>
      <th>perimeter_mean</th>
      <th>area_mean</th>
      <th>smoothness_mean</th>
      <th>compactness_mean</th>
      <th>concavity_mean</th>
      <th>concave points_mean</th>
      <th>symmetry_mean</th>
      <th>fractal_dimension_mean</th>
      <th>radius_se</th>
      <th>texture_se</th>
      <th>perimeter_se</th>
      <th>area_se</th>
      <th>smoothness_se</th>
      <th>compactness_se</th>
      <th>concavity_se</th>
      <th>concave points_se</th>
      <th>symmetry_se</th>
      <th>fractal_dimension_se</th>
      <th>radius_worst</th>
      <th>texture_worst</th>
      <th>perimeter_worst</th>
      <th>area_worst</th>
      <th>smoothness_worst</th>
      <th>compactness_worst</th>
      <th>concavity_worst</th>
      <th>concave points_worst</th>
      <th>symmetry_worst</th>
      <th>fractal_dimension_worst</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>14.127292</td>
      <td>19.289649</td>
      <td>91.969033</td>
      <td>654.889104</td>
      <td>0.096360</td>
      <td>0.104341</td>
      <td>0.088799</td>
      <td>0.048919</td>
      <td>0.181162</td>
      <td>0.062798</td>
      <td>0.405172</td>
      <td>1.216853</td>
      <td>2.866059</td>
      <td>40.337079</td>
      <td>0.007041</td>
      <td>0.025478</td>
      <td>0.031894</td>
      <td>0.011796</td>
      <td>0.020542</td>
      <td>0.003795</td>
      <td>16.269190</td>
      <td>25.677223</td>
      <td>107.261213</td>
      <td>880.583128</td>
      <td>0.132369</td>
      <td>0.254265</td>
      <td>0.272188</td>
      <td>0.114606</td>
      <td>0.290076</td>
      <td>0.083946</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.524049</td>
      <td>4.301036</td>
      <td>24.298981</td>
      <td>351.914129</td>
      <td>0.014064</td>
      <td>0.052813</td>
      <td>0.079720</td>
      <td>0.038803</td>
      <td>0.027414</td>
      <td>0.007060</td>
      <td>0.277313</td>
      <td>0.551648</td>
      <td>2.021855</td>
      <td>45.491006</td>
      <td>0.003003</td>
      <td>0.017908</td>
      <td>0.030186</td>
      <td>0.006170</td>
      <td>0.008266</td>
      <td>0.002646</td>
      <td>4.833242</td>
      <td>6.146258</td>
      <td>33.602542</td>
      <td>569.356993</td>
      <td>0.022832</td>
      <td>0.157336</td>
      <td>0.208624</td>
      <td>0.065732</td>
      <td>0.061867</td>
      <td>0.018061</td>
    </tr>
    <tr>
      <th>min</th>
      <td>6.981000</td>
      <td>9.710000</td>
      <td>43.790000</td>
      <td>143.500000</td>
      <td>0.052630</td>
      <td>0.019380</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.106000</td>
      <td>0.049960</td>
      <td>0.111500</td>
      <td>0.360200</td>
      <td>0.757000</td>
      <td>6.802000</td>
      <td>0.001713</td>
      <td>0.002252</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.007882</td>
      <td>0.000895</td>
      <td>7.930000</td>
      <td>12.020000</td>
      <td>50.410000</td>
      <td>185.200000</td>
      <td>0.071170</td>
      <td>0.027290</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.156500</td>
      <td>0.055040</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>11.700000</td>
      <td>16.170000</td>
      <td>75.170000</td>
      <td>420.300000</td>
      <td>0.086370</td>
      <td>0.064920</td>
      <td>0.029560</td>
      <td>0.020310</td>
      <td>0.161900</td>
      <td>0.057700</td>
      <td>0.232400</td>
      <td>0.833900</td>
      <td>1.606000</td>
      <td>17.850000</td>
      <td>0.005169</td>
      <td>0.013080</td>
      <td>0.015090</td>
      <td>0.007638</td>
      <td>0.015160</td>
      <td>0.002248</td>
      <td>13.010000</td>
      <td>21.080000</td>
      <td>84.110000</td>
      <td>515.300000</td>
      <td>0.116600</td>
      <td>0.147200</td>
      <td>0.114500</td>
      <td>0.064930</td>
      <td>0.250400</td>
      <td>0.071460</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>13.370000</td>
      <td>18.840000</td>
      <td>86.240000</td>
      <td>551.100000</td>
      <td>0.095870</td>
      <td>0.092630</td>
      <td>0.061540</td>
      <td>0.033500</td>
      <td>0.179200</td>
      <td>0.061540</td>
      <td>0.324200</td>
      <td>1.108000</td>
      <td>2.287000</td>
      <td>24.530000</td>
      <td>0.006380</td>
      <td>0.020450</td>
      <td>0.025890</td>
      <td>0.010930</td>
      <td>0.018730</td>
      <td>0.003187</td>
      <td>14.970000</td>
      <td>25.410000</td>
      <td>97.660000</td>
      <td>686.500000</td>
      <td>0.131300</td>
      <td>0.211900</td>
      <td>0.226700</td>
      <td>0.099930</td>
      <td>0.282200</td>
      <td>0.080040</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>15.780000</td>
      <td>21.800000</td>
      <td>104.100000</td>
      <td>782.700000</td>
      <td>0.105300</td>
      <td>0.130400</td>
      <td>0.130700</td>
      <td>0.074000</td>
      <td>0.195700</td>
      <td>0.066120</td>
      <td>0.478900</td>
      <td>1.474000</td>
      <td>3.357000</td>
      <td>45.190000</td>
      <td>0.008146</td>
      <td>0.032450</td>
      <td>0.042050</td>
      <td>0.014710</td>
      <td>0.023480</td>
      <td>0.004558</td>
      <td>18.790000</td>
      <td>29.720000</td>
      <td>125.400000</td>
      <td>1084.000000</td>
      <td>0.146000</td>
      <td>0.339100</td>
      <td>0.382900</td>
      <td>0.161400</td>
      <td>0.317900</td>
      <td>0.092080</td>
    </tr>
    <tr>
      <th>max</th>
      <td>28.110000</td>
      <td>39.280000</td>
      <td>188.500000</td>
      <td>2501.000000</td>
      <td>0.163400</td>
      <td>0.345400</td>
      <td>0.426800</td>
      <td>0.201200</td>
      <td>0.304000</td>
      <td>0.097440</td>
      <td>2.873000</td>
      <td>4.885000</td>
      <td>21.980000</td>
      <td>542.200000</td>
      <td>0.031130</td>
      <td>0.135400</td>
      <td>0.396000</td>
      <td>0.052790</td>
      <td>0.078950</td>
      <td>0.029840</td>
      <td>36.040000</td>
      <td>49.540000</td>
      <td>251.200000</td>
      <td>4254.000000</td>
      <td>0.222600</td>
      <td>1.058000</td>
      <td>1.252000</td>
      <td>0.291000</td>
      <td>0.663800</td>
      <td>0.207500</td>
    </tr>
  </tbody>
</table>
</div>



### 2.2 Unimodal Data Visualizations

One of the main goals of visualizing the data here is to observe which features are most helpful in predicting malignant or benign cancer. The other is to see general trends that may aid us in model selection and hyper parameter selection.

Apply 3 techniques that we can use to understand each attribute of your dataset independently.
* Histograms.
* Density Plots.
* Box and Whisker Plots.

#### 2.2.1. Visualise distribution of data via Histograms
Histograms are commonly used to visualize numerical variables. A histogram is similar to a bar graph after the values of the variable are grouped (binned) into a finite number of intervals (bins).

Histograms group data into bins and provide us a count of the number of observations in each bin. From the shape of the bins we can quickly get a feeling for whether an attribute is Gaussian, skewed or even has an exponential distribution. It can also help us see possible outliers.


```python
all_df.columns
```




    Index(['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
           'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
           'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
           'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
           'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
           'fractal_dimension_se', 'radius_worst', 'texture_worst',
           'perimeter_worst', 'area_worst', 'smoothness_worst',
           'compactness_worst', 'concavity_worst', 'concave points_worst',
           'symmetry_worst', 'fractal_dimension_worst'],
          dtype='object')



### Separate columns into smaller dataframes to perform visualization
Break up columns into groups, according to their suffix designation \(_mean, _se, and _worst) to perform visualization plots off. 

**Histogram the "_mean" suffix features**


```python
data_mean = all_df.iloc[:, 1:11]
data_mean.hist(bins=10, figsize=(15, 10), grid=True);
```


![png](output_24_0.png)


**Histogram the "_se" suffix features**


```python
data_mean = all_df.iloc[:, 11:21]
data_mean.hist(bins=10, figsize=(15, 10), grid=True);
```


![png](output_26_0.png)


**Histogram the "_worst" suffix features**


```python
data_mean = all_df.iloc[:, 21:]
data_mean.hist(bins=10, figsize=(15, 10), grid=True);
```


![png](output_28_0.png)


**Observation**

We can see that perhaps the attributes  **concavity** and **area** may have an exponential distribution ( ). We can also see that perhaps the **texture**, **smooth** and **symmetry** attributes may have a Gaussian or nearly Gaussian distribution. This is interesting because many machine learning techniques assume a Gaussian univariate distribution on the input variables.

#### 2.2.2. Visualise distribution of data via Density plots

**Density plots of the "_mean" suffix features**


```python
data_mean = all_df.iloc[:, 1:11]
data_mean.plot(kind='density', subplots=True, layout=(4,3), sharex=False, sharey=False, fontsize=12, figsize=(15,10));
```


![png](output_32_0.png)


**Density plots of the "_se" suffix features**


```python
data_mean = all_df.iloc[:, 11:21]
data_mean.plot(kind='density', subplots=True, layout=(4,3), sharex=False, sharey=False, fontsize=12, figsize=(15,10));
```


![png](output_34_0.png)


**Density plots of the "_worst" suffix features**


```python
all_df.iloc[:, 21:]
data_mean.plot(kind='density', subplots=True, layout=(4,3), sharex=False, sharey=False, fontsize=12, figsize=(15,10));
```


![png](output_36_0.png)


**Observation**

We can see that perhaps the attributes **perimeter**, **radius**, **area**, **concavity**, **compactness** may have an exponential distribution ( ). We can also see that perhaps the **texture**, **smooth**, **symmetry** attributes may have a Gaussian or nearly Gaussian distribution. This is interesting because many machine learning techniques assume a Gaussian univariate distribution on the input variables.

#### 2.2.3. Visualize distribution of data via Box plots

**Box plots of the "_mean" suffix features**


```python
data_mean = all_df.iloc[:, 1:11]
data_mean.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False, fontsize=12, figsize=(15,10));
```


![png](output_40_0.png)



```python
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15,10))
fig.subplots_adjust(hspace =.2, wspace=.3)
axes = axes.ravel()
for i, col in enumerate(all_df.columns[1:11]):
    _= sns.boxplot(y=col, x='diagnosis', data=all_df, ax=axes[i])
```


![png](output_41_0.png)


**Box plots of the "_se" suffix features**


```python
data_mean = all_df.iloc[:, 11:21]
data_mean.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False, fontsize=12, figsize=(15,10));
```


![png](output_43_0.png)



```python
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15,10))
fig.subplots_adjust(hspace =.2, wspace=.3)
axes = axes.ravel()
for i, col in enumerate(all_df.columns[11:21]):
    _= sns.boxplot(y=col, x='diagnosis', data=all_df, ax=axes[i])
```


![png](output_44_0.png)


**Box plots of the "_worst" suffix features**


```python
data_mean = all_df.iloc[:, 21:]
data_mean.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False, fontsize=12, figsize=(15,10));
```


![png](output_46_0.png)



```python
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15,10))
fig.subplots_adjust(hspace =.2, wspace=.3)
axes = axes.ravel()
for i, col in enumerate(all_df.columns[21:]):
    _= sns.boxplot(y=col, x='diagnosis', data=all_df, ax=axes[i])
```


![png](output_47_0.png)


**Observation**

We can see that perhaps the attributes **perimeter**, **radius**, **area**, **concavity**, **compactness** may have an exponential distribution. We can also see that perhaps the **texture**, **smooth** and **symmetry** attributes may have a Gaussian or nearly Gaussian distribution. This is interesting because many machine learning techniques assume a Gaussian univariate distribution on the input variables.

### 2.3. Multimodal Data Visualizations
* Correlation Matrix
* Scatter Plots

#### Correlation Matrix


```python
# Compute the correlation matrix
corrMatt = all_df.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corrMatt)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
fig, ax = plt.subplots(figsize=(20, 12))
plt.title('Breast Cancer Feature Correlation')

# Generate a custom diverging colormap
cmap = sns.diverging_palette(260, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corrMatt, vmax=1.2, square=False, cmap=cmap, mask=mask, ax=ax, annot=True, fmt='.2g', linewidths=1);
#sns.heatmap(corrMatt, mask=mask, vmax=1.2, square=True, annot=True, fmt='.2g', ax=ax);
```


![png](output_51_0.png)


**Observation:**

We can see strong positive relationship exists with mean values paramaters between 1 - 0.75.
* The mean area of the tissue nucleus has a strong positive correlation with mean values of radius and parameter.
* Some paramters are moderately positive correlated (r between 0.5-0.75) are concavity and area, concavity and perimeter etc.
* Likewise, we see some strong negative correlation between fractal_dimension with radius, texture, perimeter mean values.

#### Scatter Plots

**Scatter plots of the "_mean" suffix features**


```python
sns.pairplot(all_df[list(all_df.columns[1:11]) + ['diagnosis']], hue="diagnosis");
```


![png](output_55_0.png)


**Scatter plots of the "_se" suffix features**


```python
sns.pairplot(all_df[list(all_df.columns[11:21]) + ['diagnosis']], hue="diagnosis");
```


![png](output_57_0.png)


**Scatter plots of the "_worst" suffix features**


```python
sns.pairplot(all_df[list(all_df.columns[21:]) + ['diagnosis']], hue="diagnosis");
```


![png](output_59_0.png)


**Summary**

* Mean values of cell radius, perimeter, area, compactness, concavity and concave points can be used in classification of the cancer. Larger values of these parameters tends to show a correlation with malignant tumors.
* Mean values of texture, smoothness, symmetry or fractual dimension does not show a particular preference of one diagnosis over the other.
* In any of the histograms there are no noticeable large outliers that warrants further cleanup.

## 3. Pre-Processing the data

Data preprocessing is a crucial step for any data analysis problem.  It is often a very good idea to prepare our data in such way to best expose the structure of the problem to the machine learning algorithms that you intend to use.This involves a number of activities such as:
* Assigning numerical values to categorical data;
* Handling missing values; and
* Normalizing the features (so that features on small scales do not dominate when fitting a model to the data).

In the previous section we explored the data, to help gain insight on the distribution of the data as well as how the attributes correlate to each other. We identified some features of interest. Now, we will use feature selection to reduce high-dimension data, feature extraction and transformation for dimensionality reduction.


```python
all_df.columns
```




    Index(['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
           'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
           'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
           'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
           'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
           'fractal_dimension_se', 'radius_worst', 'texture_worst',
           'perimeter_worst', 'area_worst', 'smoothness_worst',
           'compactness_worst', 'concavity_worst', 'concave points_worst',
           'symmetry_worst', 'fractal_dimension_worst'],
          dtype='object')



### 3.1 Handling Categorical Attributes : Label encoding
Here, we transform the class labels from their original string representation (M and B) into integers


```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
diagnosis_encoded = encoder.fit_transform(all_df['diagnosis'])
```


```python
print(encoder.classes_)
```

    ['B' 'M']
    

After encoding the class labels(diagnosis), the malignant tumors are now represented as class 1(i.e presence of cancer cells) and the benign tumors are represented as class 0 (i.e. no cancer cells detection), respectively.

### 3.3 Feature Standardization

* Standardization is a useful technique to transform attributes with a Gaussian distribution and differing means and standard deviations to a standard Gaussian distribution with a mean of 0 and a standard deviation of 1. 

* As seen in previous exploratory section that the raw data has differing distributions which may have an impact on the most ML algorithms. Most machine learning and optimization algorithms behave much better if features are on the same scale.

Let's evaluate the same algorithms with a standardized copy of the dataset. Here, we use sklearn to scale and transform the data such that each attribute has a mean value of zero and a standard deviation of one.


```python
X = all_df.drop('diagnosis', axis=1)
y = all_df['diagnosis']
```


```python
from sklearn.preprocessing import StandardScaler

# Normalize the  data (center around 0 and scale to remove the variance).
scaler = StandardScaler()
Xs = scaler.fit_transform(X)
```

### 3.4 Feature decomposition using Principal Component Analysis( PCA)

From the pair plots in exploratory analysis section above, lot of feature pairs divide nicely the data to a similar extent, therefore, it makes sense to use one of the dimensionality reduction methods to try to use as many features as possible and retain as much information as possible when working with only 2 dimensions. We will use PCA.

Remember, PCA can be applied only on numerical data. Therefore, if the data has categorical variables they must be converted to numerical. Also, make sure we have done the basic data cleaning prior to implementing this technique. The directions of the components are identified in an unsupervised way i.e. the response variable(Y) is not used to determine the component direction. Therefore, it is an unsupervised approach and hence response variable must be removed.

Note that the PCA directions are highly sensitive to data scaling, and most likely we need to standardize the features prior to PCA if the features were measured on different scales and we want to assign equal importance to all features. Performing PCA on un-normalized variables will lead to insanely large loadings for variables with high variance. In turn, this will lead to dependence of a principal component on the variable with high variance. This is undesirable.


```python
feature_names = list(X.columns)
```


```python
from sklearn.decomposition import PCA

# dimensionality reduction
pca = PCA(n_components=10)

Xs_pca = pca.fit_transform(Xs)
```


```python
PCA_df = pd.DataFrame()
PCA_df['PCA_1'] = Xs_pca[:,0]
PCA_df['PCA_2'] = Xs_pca[:,1]
PCA_df.sample(5)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PCA_1</th>
      <th>PCA_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>64</th>
      <td>1.691608</td>
      <td>1.540677</td>
    </tr>
    <tr>
      <th>342</th>
      <td>-2.077179</td>
      <td>1.806519</td>
    </tr>
    <tr>
      <th>560</th>
      <td>-0.481771</td>
      <td>-0.178020</td>
    </tr>
    <tr>
      <th>345</th>
      <td>-2.431551</td>
      <td>3.447204</td>
    </tr>
    <tr>
      <th>143</th>
      <td>-1.837282</td>
      <td>-0.091027</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(10,6))
plt.plot(PCA_df['PCA_1'][all_df['diagnosis'] == 'M'],PCA_df['PCA_2'][all_df['diagnosis'] == 'M'],'ro', alpha = 0.7, markeredgecolor = 'k')
plt.plot(PCA_df['PCA_1'][all_df['diagnosis'] == 'B'],PCA_df['PCA_2'][all_df['diagnosis'] == 'B'],'bo', alpha = 0.7, markeredgecolor = 'k')

plt.xlabel('PCA_1')
plt.ylabel('PCA_2')
plt.legend(['Malignant','Benign']);
```


![png](output_74_0.png)


Now, what we got after applying the linear PCA transformation is a lower dimensional subspace (from 3D to 2D in this case), where the samples are "most spread" along the new feature axes.

#### Deciding How Many Principal Components to Retain

In order to decide how many principal components should be retained, it is common to summarise the results of a principal components analysis by making a **scree plot**. More about scree plot can be found [here](http://python-for-multivariate-analysis.readthedocs.io/a_little_book_of_python_for_multivariate_analysis.html), and [here](https://www.analyticsvidhya.com/blog/2016/03/practical-guide-principal-component-analysis-python/).


```python
# PCA explained variance - The amount of variance that each PC explains
var_exp = pca.explained_variance_ratio_
var_exp
```




    array([ 0.44272026,  0.18971182,  0.09393163,  0.06602135,  0.05495768,
            0.04024522,  0.02250734,  0.01588724,  0.01389649,  0.01168978])




```python
# Cumulative Variance explains
cum_var_exp = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4))
cum_var_exp
```




    array([ 0.4427,  0.6324,  0.7263,  0.7923,  0.8473,  0.8875,  0.91  ,
            0.9259,  0.9398,  0.9515])




```python
# combining above two
var_exp_ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
variance_ratios_df = pd.DataFrame(np.round(var_exp_ratios, 4), columns = ['Explained Variance'])
variance_ratios_df['Cumulative Explained Variance'] = variance_ratios_df['Explained Variance'].cumsum()

# Dimension indexing
dimensions = ['PCA_Component_{}'.format(i) for i in range(1, len(pca.components_) + 1)]

variance_ratios_df.index = dimensions
variance_ratios_df
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Explained Variance</th>
      <th>Cumulative Explained Variance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>PCA_Component_1</th>
      <td>0.4427</td>
      <td>0.4427</td>
    </tr>
    <tr>
      <th>PCA_Component_2</th>
      <td>0.1897</td>
      <td>0.6324</td>
    </tr>
    <tr>
      <th>PCA_Component_3</th>
      <td>0.0939</td>
      <td>0.7263</td>
    </tr>
    <tr>
      <th>PCA_Component_4</th>
      <td>0.0660</td>
      <td>0.7923</td>
    </tr>
    <tr>
      <th>PCA_Component_5</th>
      <td>0.0550</td>
      <td>0.8473</td>
    </tr>
    <tr>
      <th>PCA_Component_6</th>
      <td>0.0402</td>
      <td>0.8875</td>
    </tr>
    <tr>
      <th>PCA_Component_7</th>
      <td>0.0225</td>
      <td>0.9100</td>
    </tr>
    <tr>
      <th>PCA_Component_8</th>
      <td>0.0159</td>
      <td>0.9259</td>
    </tr>
    <tr>
      <th>PCA_Component_9</th>
      <td>0.0139</td>
      <td>0.9398</td>
    </tr>
    <tr>
      <th>PCA_Component_10</th>
      <td>0.0117</td>
      <td>0.9515</td>
    </tr>
  </tbody>
</table>
</div>



**Scree Plot**


```python
plt.figure(figsize=(8,6))
plt.bar(range(1, len(pca.components_) + 1), var_exp, alpha=0.5, align='center', label='individual explained variance')
plt.step(range(1, len(pca.components_) + 1), cum_var_exp, where='mid', label='cumulative explained variance')
plt.plot(range(1, len(pca.components_) + 1), var_exp, 'ro-')
plt.xticks(range(1, len(pca.components_) + 1))
plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal Components')
plt.title('Scree Plot')
plt.legend(loc='best');
```


![png](output_81_0.png)


**Observation**

The most obvious change in slope in the scree plot occurs at component 2, which is the `"elbow"` of the scree plot. Therefore, it cound be argued based on the basis of the scree plot that the first three components should be retained.

#### Principal Components Feature Weights as function of the components: Bar Plot


```python
# PCA components
components_df = pd.DataFrame(np.round(pca.components_, 4), columns = feature_names)

# Dimension indexing
dimensions = ['PCA_{}'.format(i) for i in range(1, len(pca.components_) + 1)]

components_df.index = dimensions
components_df
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>radius_mean</th>
      <th>texture_mean</th>
      <th>perimeter_mean</th>
      <th>area_mean</th>
      <th>smoothness_mean</th>
      <th>compactness_mean</th>
      <th>concavity_mean</th>
      <th>concave points_mean</th>
      <th>symmetry_mean</th>
      <th>fractal_dimension_mean</th>
      <th>radius_se</th>
      <th>texture_se</th>
      <th>perimeter_se</th>
      <th>area_se</th>
      <th>smoothness_se</th>
      <th>compactness_se</th>
      <th>concavity_se</th>
      <th>concave points_se</th>
      <th>symmetry_se</th>
      <th>fractal_dimension_se</th>
      <th>radius_worst</th>
      <th>texture_worst</th>
      <th>perimeter_worst</th>
      <th>area_worst</th>
      <th>smoothness_worst</th>
      <th>compactness_worst</th>
      <th>concavity_worst</th>
      <th>concave points_worst</th>
      <th>symmetry_worst</th>
      <th>fractal_dimension_worst</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>PCA_1</th>
      <td>0.2189</td>
      <td>0.1037</td>
      <td>0.2275</td>
      <td>0.2210</td>
      <td>0.1426</td>
      <td>0.2393</td>
      <td>0.2584</td>
      <td>0.2609</td>
      <td>0.1382</td>
      <td>0.0644</td>
      <td>0.2060</td>
      <td>0.0174</td>
      <td>0.2113</td>
      <td>0.2029</td>
      <td>0.0145</td>
      <td>0.1704</td>
      <td>0.1536</td>
      <td>0.1834</td>
      <td>0.0425</td>
      <td>0.1026</td>
      <td>0.2280</td>
      <td>0.1045</td>
      <td>0.2366</td>
      <td>0.2249</td>
      <td>0.1280</td>
      <td>0.2101</td>
      <td>0.2288</td>
      <td>0.2509</td>
      <td>0.1229</td>
      <td>0.1318</td>
    </tr>
    <tr>
      <th>PCA_2</th>
      <td>-0.2339</td>
      <td>-0.0597</td>
      <td>-0.2152</td>
      <td>-0.2311</td>
      <td>0.1861</td>
      <td>0.1519</td>
      <td>0.0602</td>
      <td>-0.0348</td>
      <td>0.1903</td>
      <td>0.3666</td>
      <td>-0.1056</td>
      <td>0.0900</td>
      <td>-0.0895</td>
      <td>-0.1523</td>
      <td>0.2044</td>
      <td>0.2327</td>
      <td>0.1972</td>
      <td>0.1303</td>
      <td>0.1838</td>
      <td>0.2801</td>
      <td>-0.2199</td>
      <td>-0.0455</td>
      <td>-0.1999</td>
      <td>-0.2194</td>
      <td>0.1723</td>
      <td>0.1436</td>
      <td>0.0980</td>
      <td>-0.0083</td>
      <td>0.1419</td>
      <td>0.2753</td>
    </tr>
    <tr>
      <th>PCA_3</th>
      <td>-0.0085</td>
      <td>0.0645</td>
      <td>-0.0093</td>
      <td>0.0287</td>
      <td>-0.1043</td>
      <td>-0.0741</td>
      <td>0.0027</td>
      <td>-0.0256</td>
      <td>-0.0402</td>
      <td>-0.0226</td>
      <td>0.2685</td>
      <td>0.3746</td>
      <td>0.2666</td>
      <td>0.2160</td>
      <td>0.3088</td>
      <td>0.1548</td>
      <td>0.1765</td>
      <td>0.2247</td>
      <td>0.2886</td>
      <td>0.2115</td>
      <td>-0.0475</td>
      <td>-0.0423</td>
      <td>-0.0485</td>
      <td>-0.0119</td>
      <td>-0.2598</td>
      <td>-0.2361</td>
      <td>-0.1731</td>
      <td>-0.1703</td>
      <td>-0.2713</td>
      <td>-0.2328</td>
    </tr>
    <tr>
      <th>PCA_4</th>
      <td>0.0414</td>
      <td>-0.6030</td>
      <td>0.0420</td>
      <td>0.0534</td>
      <td>0.1594</td>
      <td>0.0318</td>
      <td>0.0191</td>
      <td>0.0653</td>
      <td>0.0671</td>
      <td>0.0486</td>
      <td>0.0979</td>
      <td>-0.3599</td>
      <td>0.0890</td>
      <td>0.1082</td>
      <td>0.0447</td>
      <td>-0.0275</td>
      <td>0.0013</td>
      <td>0.0741</td>
      <td>0.0441</td>
      <td>0.0153</td>
      <td>0.0154</td>
      <td>-0.6328</td>
      <td>0.0138</td>
      <td>0.0259</td>
      <td>0.0177</td>
      <td>-0.0913</td>
      <td>-0.0740</td>
      <td>0.0060</td>
      <td>-0.0363</td>
      <td>-0.0771</td>
    </tr>
    <tr>
      <th>PCA_5</th>
      <td>0.0378</td>
      <td>-0.0495</td>
      <td>0.0374</td>
      <td>0.0103</td>
      <td>-0.3651</td>
      <td>0.0117</td>
      <td>0.0864</td>
      <td>-0.0439</td>
      <td>-0.3059</td>
      <td>-0.0444</td>
      <td>-0.1545</td>
      <td>-0.1917</td>
      <td>-0.1210</td>
      <td>-0.1276</td>
      <td>-0.2321</td>
      <td>0.2800</td>
      <td>0.3540</td>
      <td>0.1955</td>
      <td>-0.2529</td>
      <td>0.2633</td>
      <td>-0.0044</td>
      <td>-0.0929</td>
      <td>0.0075</td>
      <td>-0.0274</td>
      <td>-0.3244</td>
      <td>0.1218</td>
      <td>0.1885</td>
      <td>0.0433</td>
      <td>-0.2446</td>
      <td>0.0944</td>
    </tr>
    <tr>
      <th>PCA_6</th>
      <td>0.0187</td>
      <td>-0.0322</td>
      <td>0.0173</td>
      <td>-0.0019</td>
      <td>-0.2864</td>
      <td>-0.0141</td>
      <td>-0.0093</td>
      <td>-0.0520</td>
      <td>0.3565</td>
      <td>-0.1194</td>
      <td>-0.0256</td>
      <td>-0.0287</td>
      <td>0.0018</td>
      <td>-0.0429</td>
      <td>-0.3429</td>
      <td>0.0692</td>
      <td>0.0563</td>
      <td>-0.0312</td>
      <td>0.4902</td>
      <td>-0.0532</td>
      <td>-0.0003</td>
      <td>-0.0500</td>
      <td>0.0085</td>
      <td>-0.0252</td>
      <td>-0.3693</td>
      <td>0.0477</td>
      <td>0.0284</td>
      <td>-0.0309</td>
      <td>0.4989</td>
      <td>-0.0802</td>
    </tr>
    <tr>
      <th>PCA_7</th>
      <td>-0.1241</td>
      <td>0.0114</td>
      <td>-0.1145</td>
      <td>-0.0517</td>
      <td>-0.1407</td>
      <td>0.0309</td>
      <td>-0.1075</td>
      <td>-0.1505</td>
      <td>-0.0939</td>
      <td>0.2958</td>
      <td>0.3125</td>
      <td>-0.0908</td>
      <td>0.3146</td>
      <td>0.3467</td>
      <td>-0.2440</td>
      <td>0.0235</td>
      <td>-0.2088</td>
      <td>-0.3696</td>
      <td>-0.0804</td>
      <td>0.1914</td>
      <td>-0.0097</td>
      <td>0.0099</td>
      <td>-0.0004</td>
      <td>0.0678</td>
      <td>-0.1088</td>
      <td>0.1405</td>
      <td>-0.0605</td>
      <td>-0.1680</td>
      <td>-0.0185</td>
      <td>0.3747</td>
    </tr>
    <tr>
      <th>PCA_8</th>
      <td>-0.0075</td>
      <td>0.1307</td>
      <td>-0.0187</td>
      <td>0.0347</td>
      <td>-0.2890</td>
      <td>-0.1514</td>
      <td>-0.0728</td>
      <td>-0.1523</td>
      <td>-0.2315</td>
      <td>-0.1771</td>
      <td>0.0225</td>
      <td>-0.4754</td>
      <td>-0.0119</td>
      <td>0.0858</td>
      <td>0.5734</td>
      <td>0.1175</td>
      <td>0.0606</td>
      <td>-0.1083</td>
      <td>0.2201</td>
      <td>0.0112</td>
      <td>0.0426</td>
      <td>0.0363</td>
      <td>0.0306</td>
      <td>0.0794</td>
      <td>0.2059</td>
      <td>0.0840</td>
      <td>0.0725</td>
      <td>-0.0362</td>
      <td>0.2282</td>
      <td>0.0484</td>
    </tr>
    <tr>
      <th>PCA_9</th>
      <td>-0.2231</td>
      <td>0.1127</td>
      <td>-0.2237</td>
      <td>-0.1956</td>
      <td>0.0064</td>
      <td>-0.1678</td>
      <td>0.0406</td>
      <td>-0.1120</td>
      <td>0.2560</td>
      <td>-0.1237</td>
      <td>0.2500</td>
      <td>-0.2466</td>
      <td>0.2272</td>
      <td>0.2292</td>
      <td>-0.1419</td>
      <td>-0.1453</td>
      <td>0.3581</td>
      <td>0.2725</td>
      <td>-0.3041</td>
      <td>-0.2137</td>
      <td>-0.1121</td>
      <td>0.1033</td>
      <td>-0.1096</td>
      <td>-0.0807</td>
      <td>0.1123</td>
      <td>-0.1007</td>
      <td>0.1619</td>
      <td>0.0605</td>
      <td>0.0646</td>
      <td>-0.1342</td>
    </tr>
    <tr>
      <th>PCA_10</th>
      <td>0.0955</td>
      <td>0.2409</td>
      <td>0.0864</td>
      <td>0.0750</td>
      <td>-0.0693</td>
      <td>0.0129</td>
      <td>-0.1356</td>
      <td>0.0081</td>
      <td>0.5721</td>
      <td>0.0811</td>
      <td>-0.0495</td>
      <td>-0.2891</td>
      <td>-0.1145</td>
      <td>-0.0919</td>
      <td>0.1609</td>
      <td>0.0435</td>
      <td>-0.1413</td>
      <td>0.0862</td>
      <td>-0.3165</td>
      <td>0.3675</td>
      <td>0.0774</td>
      <td>0.0295</td>
      <td>0.0505</td>
      <td>0.0699</td>
      <td>-0.1283</td>
      <td>-0.1721</td>
      <td>-0.3116</td>
      <td>-0.0766</td>
      <td>-0.0296</td>
      <td>0.0126</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plot the feature weights as a function of the components
# Create a bar plot visualization
fig, ax = plt.subplots(figsize = (20, 10))
    
_ = components_df.plot(ax = ax, kind = 'bar')
_ = ax.set_ylabel("Feature Weights")
_ = ax.set_xticklabels(dimensions, rotation=0)

# Display the explained variance ratios
for i, ev in enumerate(pca.explained_variance_ratio_):
    _ = ax.text(i-0.40, ax.get_ylim()[1] + 0.05, "Explained Variance\n %.4f"%(ev))
```


![png](output_85_0.png)


#### Principal Components Feature Weights as function of the components: HeatMap


```python
# vizualizing principle components as a heatmap this allows us to see what dimensions in the 'original space' are active
plt.matshow(pca.components_[0:3], cmap='viridis')
plt.yticks([0, 1, 2], ["PCA_1", "PCA_2", "PCA_3"])
plt.colorbar()
plt.xticks(range(len(feature_names)), feature_names, rotation=60, ha='left')
plt.xlabel("Feature")
plt.ylabel("Principal components");
```


![png](output_87_0.png)


### Visualizing a Biplot : Principal Components Loadings
A biplot is a scatterplot where each data point is represented by its scores along the principal components. The axes are the principal components (in this case `PCA_1` and `PCA_2`). In addition, the biplot shows the projection of the original features along the components. A biplot can help us interpret the reduced dimensions of the data, and discover relationships between the principal components and original features.


```python
# Apply PCA by fitting the scaled data with only two dimensions
from sklearn.decomposition import PCA
pca = PCA(n_components=2).fit(Xs)

# Transform the original data using the PCA fit above
Xs_pca = pca.transform(Xs)

# Create a DataFrame for the pca transformed data
Xs_df = pd.DataFrame(Xs, columns=feature_names)

# Create a DataFrame for the pca transformed data
Xs_pca_df = pd.DataFrame(Xs_pca, columns = ['PCA_1', 'PCA_2'])
```


```python
pca.components_.T
```




    array([[ 0.21890244, -0.23385713],
           [ 0.10372458, -0.05970609],
           [ 0.22753729, -0.21518136],
           [ 0.22099499, -0.23107671],
           [ 0.14258969,  0.18611302],
           [ 0.23928535,  0.15189161],
           [ 0.25840048,  0.06016536],
           [ 0.26085376, -0.0347675 ],
           [ 0.13816696,  0.19034877],
           [ 0.06436335,  0.36657547],
           [ 0.20597878, -0.10555215],
           [ 0.01742803,  0.08997968],
           [ 0.21132592, -0.08945723],
           [ 0.20286964, -0.15229263],
           [ 0.01453145,  0.20443045],
           [ 0.17039345,  0.2327159 ],
           [ 0.15358979,  0.19720728],
           [ 0.1834174 ,  0.13032156],
           [ 0.04249842,  0.183848  ],
           [ 0.10256832,  0.28009203],
           [ 0.22799663, -0.21986638],
           [ 0.10446933, -0.0454673 ],
           [ 0.23663968, -0.19987843],
           [ 0.22487053, -0.21935186],
           [ 0.12795256,  0.17230435],
           [ 0.21009588,  0.14359317],
           [ 0.22876753,  0.09796411],
           [ 0.25088597, -0.00825724],
           [ 0.12290456,  0.14188335],
           [ 0.13178394,  0.27533947]])




```python
Xs.shape, pca.components_.T.shape
```




    ((569, 30), (30, 2))




```python
# Create a biplot
def biplot(original_data, reduced_data, pca):
    '''
    Produce a biplot that shows a scatterplot of the reduced
    data and the projections of the original features.
    
    original_data: original data, before transformation.
               Needs to be a pandas dataframe with valid column names
    reduced_data: the reduced data (the first two dimensions are plotted)
    pca: pca object that contains the components_ attribute

    return: a matplotlib AxesSubplot object (for any additional customization)
    
    This procedure is inspired by the script:
    https://github.com/teddyroland/python-biplot
    '''

    fig, ax = plt.subplots(figsize = (20, 15))
    # scatterplot of the reduced data    
    ax.scatter(x=reduced_data.loc[:, 'PCA_1'], y=reduced_data.loc[:, 'PCA_2'], 
        facecolors='b', edgecolors='b', s=70, alpha=0.5)
    
    feature_vectors = pca.components_.T

    # we use scaling factors to make the arrows easier to see
    arrow_size, text_pos = 30.0, 32.0,

    # projections of the original features
    for i, v in enumerate(feature_vectors):
        ax.arrow(0, 0, arrow_size*v[0], arrow_size*v[1], 
                  head_width=0.2, head_length=0.2, linewidth=1, color='red')
        ax.text(v[0]*text_pos, v[1]*text_pos, original_data.columns[i], color='black', 
                 ha='center', va='center', fontsize=9)

    ax.set_xlabel("PCA_1", fontsize=14)
    ax.set_ylabel("PCA_2", fontsize=14)
    ax.set_title("PC plane with original feature projections.", fontsize=16);
    return ax
```


```python
biplot(Xs_df, Xs_pca_df, pca);
```


![png](output_93_0.png)


#### Principal Components Important Points

- We should not combine the train and test set to obtain PCA components of whole data at once. Because, this would violate the entire assumption of generalization since test data would get 'leaked' into the training set. In other words, the test data set would no longer remain 'unseen'. Eventually, this will hammer down the generalization capability of the model.

- We should not perform PCA on test and train data sets separately. Because, the resultant vectors from train and test PCAs will have different directions ( due to unequal variance). Due to this, we’ll end up comparing data registered on different axes. Therefore, the resulting vectors from train and test data should have same axes.

- We should do exactly the same transformation to the test set as we did to training set, including the center and scaling feature.

### 3.5 Feature decomposition using t-SNE

t-Distributed Stochastic Neighbor Embedding ([t-SNE](http://lvdmaaten.github.io/tsne/)) is another technique for dimensionality reduction and is particularly well suited for the visualization of high-dimensional datasets. Contrary to PCA it is not a mathematical technique but a probablistic one. The original paper describes the working of t-SNE as:

*t-Distributed stochastic neighbor embedding (t-SNE) minimizes the divergence between two distributions: a distribution that measures pairwise similarities of the input objects and a distribution that measures pairwise similarities of the corresponding low-dimensional points in the embedding.*

Essentially what this means is that it looks at the original data that is entered into the algorithm and looks at how to best represent this data using less dimensions by matching both distributions. The way it does this is computationally quite heavy and therefore there are some (serious) limitations to the use of this technique. For example one of the recommendations is that, in case of very high dimensional data, you may need to apply another dimensionality reduction technique before using t-SNE:

*It is highly recommended to use another dimensionality reduction method (e.g. PCA for dense data or TruncatedSVD for sparse data) to reduce the number of dimensions to a reasonable amount (e.g. 50) if the number of features is very high.*

The other key drawback is that since t-SNE scales quadratically in the number of objects N, its applicability is limited to data sets with only a few thousand input objects; beyond that, learning becomes too slow to be practical (and the memory requirements become too large).


```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=rnd_seed)
Xs_tsne = tsne.fit_transform(Xs)
```


```python
plt.figure(figsize=(8,6))
plt.plot(Xs_tsne[y == 'M'][:, 0], Xs_tsne[y == 'M'][:, 1], 'ro', alpha = 0.7, markeredgecolor='k')
plt.plot(Xs_tsne[y == 'B'][:, 0], Xs_tsne[y == 'B'][:, 1], 'bo', alpha = 0.7, markeredgecolor='k')
plt.legend(['Malignant','Benign']);
```


![png](output_97_0.png)


It is common to select a subset of features that have the largest correlation with the class labels. The effect of feature selection must be assessed within a complete modeling pipeline in order to give us an unbiased estimated of our model's true performance. Hence, in the next section we will use cross-validation, before applying the PCA-based feature selection strategy in the model building pipeline.

## 4. Predictive model using Support Vector Machine (SVM)

Support vector machines (SVMs) learning algorithm will be used to build the predictive model.  SVMs are one of the most popular classification algorithms, and have an elegant way of transforming nonlinear data so that one can use a linear algorithm to fit a linear model to the data (Cortes and Vapnik 1995)

Kernelized support vector machines are powerful models and perform well on a variety of datasets. 
1. SVMs allow for complex decision boundaries, even if the data has only a few features. 
2. They work well on low-dimensional and high-dimensional data (i.e., few and many features), but don’t scale very well with the number of samples.
    **Running an SVM on data with up to 10,000 samples might work well, but working with datasets of size 100,000 or more can become challenging in terms of runtime and memory usage.**

3. SVMs requires careful preprocessing of the data and tuning of the parameters. This is why, these days, most people instead use tree-based models such as random forests or gradient boosting (which require little or no preprocessing) in many applications.

4.  SVM models are hard to inspect; it can be difficult to understand why a particular prediction was made, and it might be tricky to explain the model to a nonexpert.

#### Important Parameters
The important parameters in kernel SVMs are the
* Regularization parameter C; 
* The choice of the kernel - (linear, radial basis function(RBF) or polynomial);
* Kernel-specific parameters. 

gamma and C both control the complexity of the model, with large values in either resulting in a more complex model. Therefore, good settings for the two parameters are usually strongly correlated, and C and gamma should be adjusted together.

### Split data into training and test sets

The simplest method to evaluate the performance of a machine learning algorithm is to use different training and testing datasets. splitting the data into test and training sets is crucial to avoid overfitting. This allows generalization of real, previously-unseen data. Here we will
* split the available data into a training set and a testing set (70% training, 30% test)
* train the algorithm on the first part
* make predictions on the second part and 
* evaluate the predictions against the expected results

The size of the split can depend on the size and specifics of our dataset, although it is common to use 67% of the data for training and the remaining 33% for testing.


```python
from sklearn.preprocessing import LabelEncoder

# transform the class labels from their original string representation (M and B) into integers
le = LabelEncoder()
all_df['diagnosis'] = le.fit_transform(all_df['diagnosis'])
```


```python
X = all_df.drop('diagnosis', axis=1) # drop labels for training set
y = all_df['diagnosis']
```


```python
# # stratified sampling. Divide records in training and testing sets.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=rnd_seed, stratify=y)
```


```python
# Normalize the  data (center around 0 and scale to remove the variance).
scaler = StandardScaler()
Xs_train = scaler.fit_transform(X_train)
```


```python
# Create an SVM classifier and train it on 70% of the data set.
from sklearn.svm import SVC

clf = SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', probability=True)
clf.fit(Xs_train, y_train)
```




    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=True, random_state=None, shrinking=True,
      tol=0.001, verbose=False)




```python
Xs_test = scaler.transform(X_test)
```


```python
classifier_score = clf.score(Xs_test, y_test)
```


```python
print('The classifier accuracy score is {:03.2f}'.format(classifier_score))
```

    The classifier accuracy score is 0.99
    

To get a better measure of prediction accuracy (which we can use as a proxy for "goodness of fit" of the model), we can successively split the data into folds that we will use for training and testing:

### Classification with cross-validation

Cross-validation extends the idea of train and test set split idea further. Instead of having a single train/test split, we specify **folds** so that the data is divided into similarly-sized folds. 

* Training occurs by taking all folds except one - referred to as the holdout sample.

* On the completion of the training, we test the performance of our fitted model using the holdout sample. 

* The holdout sample is then thrown back with the rest of the other folds, and a different fold is pulled out as the new holdout sample. 

* Training is repeated again with the remaining folds and we measure performance using the holdout sample. This process is repeated until each fold has had a chance to be a test or holdout sample. 

* The expected performance of the classifier, called cross-validation error, is then simply an average of error rates computed on each holdout sample. 

This process is demonstrated by first performing a standard train/test split, and then computing cross-validation error.


```python
# Get average of 3-fold cross-validation score using an SVC estimator.
from sklearn.model_selection import cross_val_score
n_folds = 3
clf_cv = SVC(C=1.0, kernel='rbf', degree=3, gamma='auto')
cv_error = np.average(cross_val_score(clf_cv, Xs_train, y_train, cv=n_folds))
```


```python
print('The {}-fold cross-validation accuracy score for this classifier is {:.2f}'.format(n_folds, cv_error))
```

    The 3-fold cross-validation accuracy score for this classifier is 0.96
    

### Classification with Feature Selection & cross-validation

The above evaluations were based on using the entire set of features. We will now employ the correlation-based feature selection strategy to assess the effect of using 3 features which have the best correlation with the class labels.


```python
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline

# model with just 3 features selected
clf_fs_cv = Pipeline([
    ('feature_selector', SelectKBest(f_classif, k=3)),
    ('svc', SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', probability=True))
])

scores = cross_val_score(clf_fs_cv, Xs_train, y_train, cv=3)
```


```python
print(scores)
avg = (100 * np.mean(scores), 100 * np.std(scores)/np.sqrt(scores.shape[0]))
print("Average score and uncertainty: (%.2f +- %.3f)%%"  %avg)
```

    [ 0.93283582  0.93939394  0.9469697 ]
    Average score and uncertainty: (93.97 +- 0.333)%
    

From the above results, we can see that only a fraction of the features are required to build a model that performs similarly to models based on using the entire set of features.

Feature selection is an important part of the model-building process that we must always pay particular attention to. The details are beyond the scope of this notebook. In the rest of the analysis, we will continue using the entire set of features.

### Model Accuracy: Receiver Operating Characteristic (ROC) curve

In statistical modeling and machine learning, a commonly-reported performance measure of model accuracy for binary classification problems is Area Under the Curve (AUC).

To understand what information the ROC curve conveys, consider the so-called confusion matrix that essentially is a two-dimensional table where the classifier model is on one axis (vertical), and ground truth is on the other (horizontal) axis, as shown below. Either of these axes can take two values (as depicted)


|                 |Model predicts "+" |Model predicts  "-" |
|---------------- | ----------------- | -------------------|
|** Actual: "+" **| `True positive`   | `False negative`   | 
|** Actual: "-" **| `False positive`  | `True negative`    |
 In an ROC curve, we plot "True Positive Rate" on the Y-axis and "False Positive Rate" on the X-axis, where the values "true positive", "false negative", "false positive", and "true negative" are events (or their probabilities) as described above. The rates are defined according to the following:
* True positive rate (or sensitivity): tpr = tp / (tp + fn)
* False positive rate:       fpr = fp / (fp + tn)
* True negative rate (or specificity): tnr = tn / (fp + tn)

In all definitions, the denominator is a row margin in the above confusion matrix. Thus,one can  express
* the true positive rate (tpr) as the probability that the model says "+" when the real value is indeed "+" (i.e., a conditional probability). However, this does not tell us how likely we are to be correct when calling "+" (i.e., the probability of a true positive, conditioned on the test result being "+").          


```python
# The confusion matrix helps visualize the performance of the algorithm.
from sklearn.metrics import confusion_matrix, classification_report

y_pred = clf.fit(Xs_train, y_train).predict(Xs_test)
cm = confusion_matrix(y_test, y_pred)
```


```python
# lengthy way to plot confusion matrix, a shorter way using seaborn is also shown somewhere downa
fig, ax = plt.subplots(figsize=(3, 3))
ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
for i in range(cm.shape[0]):
     for j in range(cm.shape[1]):
         ax.text(x=j, y=i,
                s=cm[i, j], 
                va='center', ha='center')
classes=["Benign","Malignant"]
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.xlabel('Predicted Values', )
plt.ylabel('Actual Values');
```


![png](output_120_0.png)



```python
print(classification_report(y_test, y_pred ))
```

                 precision    recall  f1-score   support
    
              0       1.00      0.98      0.99       107
              1       0.97      1.00      0.98        64
    
    avg / total       0.99      0.99      0.99       171
    
    

#### Observation 
There are two possible predicted classes: "1" and "0". Malignant = 1 (indicates prescence of cancer cells) and Benign
= 0 (indicates abscence).

* The classifier made a total of 171 predictions (i.e 171 patients were being tested for the presence breast cancer).
* Out of those 171 cases, the classifier predicted "yes" 66 times, and "no" 105 times.
* In reality, 64 patients in the sample have the disease, and 107 patients do not.

#### Rates as computed from the confusion matrix
1. **Accuracy**: Overall, how often is the classifier correct?
    * (TP+TN)/total = (TP+TN)/(P+N) = (64 + 105)/171 = 0.98

2. **Misclassification Rate**: Overall, how often is it wrong?
    * (FP+FN)/total = (FP+FN)/(P+N) = (2 + 0)/171 = 0.011 equivalent to 1 minus Accuracy also known as ***"Error Rate"***

3. **True Positive Rate:** When it's actually yes, how often does it predict 1? Out of all the positive (majority class) values, how many have been predicted correctly
   * TP/actual yes = TP/(TP + FN) = 64/(64 + 0) = 1.00 also known as ***"Sensitivity"*** or ***"Recall"***

4. **False Positive Rate**: When it's actually 0, how often does it predict 1?
   * FP/actual no = FP/N = FP/(FP + TN) = 2/(2 + 105) = 0.018 equivalent to 1 minus true negative rate

5. **True Negative Rate**: When it's actually 0, how often does it predict 0? Out of all the negative (minority class) values, how many have been predicted correctly’
   * TN/actual no = TN / N = TN/(TN + FP) = 105/(105 + 2) = 0.98 also known as ***Specificity***, equivalent to 1 minus False Positive Rate

6. **Precision**: When it predicts 1, how often is it correct?
   * TP/predicted yes = TP/(TP + FP) = 64/(64 + 2) = 0.97

7. **Prevalence**: How often does the yes condition actually occur in our sample?
   * actual yes/total = 64/171 = 0.34

8. **F score**: It is the harmonic mean of precision and recall. It is used to compare several models side-by-side. Higher the better.
   * 2 x (Precision x Recall)/ (Precision + Recall)  = 2 x (0.97 x 1.00) / (0.97 + 1.00) = 0.98 


```python
from sklearn.metrics import roc_curve, auc
# Plot the receiver operating characteristic curve (ROC).
plt.figure(figsize=(10,8))
probas_ = clf.predict_proba(Xs_test)
fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, lw=1, label='ROC fold (area = %0.2f)' % (roc_auc))
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate | 1 - specificity (1 - Benign recall)')
plt.ylabel('True Positive Rate | Sensitivity (Malignant recall)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.axes().set_aspect(1);
```


![png](output_123_0.png)


* To interpret the ROC correctly, consider what the points that lie along the diagonal represent. For these situations, there is an equal chance of "+" and "-" happening. Therefore, this is not that different from making a prediction by tossing of an unbiased coin. Put simply, the classification model is random.

* For the points above the diagonal, tpr > fpr, and the model says that we are in a zone where we are performing better than random. For example, assume tpr = 0.99 and fpr = 0.01, Then, the probability of being in the true positive group is (0.99 / (0.99 + 0.01)) = 99. Furthermore, holding fpr constant, it is easy to see that the more vertically above the diagonal we are positioned, the better the classification model.

## 5. Optimizing the SVM Classifier

Machine learning models are parameterized so that their behavior can be tuned for a given problem. Models can have many parameters and finding the best combination of parameters can be treated as a search problem.

### 5.1 Importance of optimizing a classifier

We can tune two key parameters of the SVM algorithm:
* the value of C (how much to relax the margin) 
* and the type of kernel. 

The default for SVM (the SVC class) is to use the Radial Basis Function (RBF) kernel with a C value set to 1.0. We will perform a grid search using 5-fold cross validation with a standardized copy of the training dataset. We will try a number of simpler kernel types and C values with less bias and more bias (less than and more than 1.0 respectively).

Python scikit-learn provides two simple methods for algorithm parameter tuning:
* Grid Search Parameter Tuning. 
* Random Search Parameter Tuning.


```python
from sklearn.model_selection import GridSearchCV

# Train classifiers.
kernel_values = [ 'linear', 'poly', 'rbf', 'sigmoid' ]
param_grid = {'C': np.logspace(-3, 2, 6), 'gamma': np.logspace(-3, 2, 6), 'kernel': kernel_values}

grid = GridSearchCV(SVC(), param_grid=param_grid, cv=5)
grid.fit(Xs_train, y_train)
```




    GridSearchCV(cv=5, error_score='raise',
           estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False),
           fit_params=None, iid=True, n_jobs=1,
           param_grid={'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'gamma': array([  1.00000e-03,   1.00000e-02,   1.00000e-01,   1.00000e+00,
             1.00000e+01,   1.00000e+02]), 'C': array([  1.00000e-03,   1.00000e-02,   1.00000e-01,   1.00000e+00,
             1.00000e+01,   1.00000e+02])},
           pre_dispatch='2*n_jobs', refit=True, return_train_score=True,
           scoring=None, verbose=0)




```python
print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))
```

    The best parameters are {'kernel': 'rbf', 'gamma': 0.001, 'C': 10.0} with a score of 0.97
    


```python
best_clf = grid.best_estimator_
best_clf.probability = True
```


```python
y_pred = best_clf.fit(Xs_train, y_train).predict(Xs_test)
cm = confusion_matrix(y_test, y_pred)
```


```python
# using seaborn to plot confusion matrix
classes=["Benign","Malignant"]
df_cm = pd.DataFrame(cm, index=classes, columns=classes)
fig = plt.figure(figsize=(3,3))
ax = sns.heatmap(df_cm, annot=True, fmt="d")
ax.yaxis.set_ticklabels(ax.yaxis.get_ticklabels(), rotation=0, ha='right')
ax.xaxis.set_ticklabels(ax.xaxis.get_ticklabels(), rotation=45, ha='right')
plt.xlabel('Predicted Values', )
plt.ylabel('Actual Values');
```


![png](output_131_0.png)



```python
print(classification_report(y_test, y_pred ))
```

                 precision    recall  f1-score   support
    
              0       0.96      1.00      0.98       107
              1       1.00      0.94      0.97        64
    
    avg / total       0.98      0.98      0.98       171
    
    

### 5.2 Visualizing the SVM Boundary

Based on the best classifier that we got from our optimization process we would now try to visualize the decision boundary of the SVM. In order to visualize the SVM decision boundary we need to reduce the multi-dimensional data to two dimension. We will resort to applying the linear PCA transformation that will transofrm our data to a lower dimensional subspace (from 30D to 2D in this case).


```python
# Apply PCA by fitting the scaled data with only two dimensions
from sklearn.decomposition import PCA
pca = PCA(n_components=2)

# Transform the original data using the PCA fit above
Xs_train_pca = pca.fit_transform(Xs_train)
```


```python
# Take the first two PCA features. We could avoid this by using a two-dim dataset
X = Xs_train_pca
y = y_train
```


```python
# http://scikit-learn.org/stable/auto_examples/svm/plot_iris.html
def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy
```


```python
# http://scikit-learn.org/stable/auto_examples/svm/plot_iris.html
def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out
```


```python
# create a mesh of values from the 1st two PCA components
X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)
```


```python
clf = SVC(C=10.0, kernel='rbf', gamma=0.001)
clf.fit(X, y)
```




    SVC(C=10.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)




```python
fig, ax = plt.subplots(figsize=(12,6))
plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.4)
plt.plot(Xs_train_pca[y_train == 1][:, 0],Xs_train_pca[y_train == 1][:, 1], 'ro', alpha=0.8, markeredgecolor='k', label='Malignant')
plt.plot(Xs_train_pca[y_train == 0][:, 0],Xs_train_pca[y_train == 0][:, 1], 'bo', alpha=0.8, markeredgecolor='k', label='Benign')

svs = clf.support_vectors_
plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#00AD00', edgecolors='k', label='Support Vectors')
    
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xlabel('PCA_1')
plt.ylabel('PCA_2')
plt.title('Decision Boundary of SVC with RBF kernel')
plt.legend();
```


![png](output_140_0.png)


## 6. Automate the ML process using pipelines 

There are standard workflows in a machine learning project that can be automated. In Python `scikit-learn`, Pipelines help to clearly define and automate these workflows.
* Pipelines help overcome common problems like data leakage in our test harness. 
* Python scikit-learn provides a Pipeline utility to help automate machine learning workflows.
* Pipelines work by allowing for a linear sequence of data transforms to be chained together culminating in a modeling process that can be evaluated.

### 6.1 Data Preparation and Modeling Pipeline

####  Evaluate Some Algorithms
Now it is time to create some models of the data and estimate their accuracy on unseen data. Here is what we are going to cover in this step:
1. Separate out a validation dataset.
2. Setup the test harness to use 10-fold cross validation.
3. Build 5 different models  
4. Select the best model

#### Validation Dataset


```python
# read the data
all_df = pd.read_csv('data/data.csv', index_col=False)
all_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>diagnosis</th>
      <th>radius_mean</th>
      <th>texture_mean</th>
      <th>perimeter_mean</th>
      <th>area_mean</th>
      <th>smoothness_mean</th>
      <th>compactness_mean</th>
      <th>concavity_mean</th>
      <th>concave points_mean</th>
      <th>symmetry_mean</th>
      <th>fractal_dimension_mean</th>
      <th>radius_se</th>
      <th>texture_se</th>
      <th>perimeter_se</th>
      <th>area_se</th>
      <th>smoothness_se</th>
      <th>compactness_se</th>
      <th>concavity_se</th>
      <th>concave points_se</th>
      <th>symmetry_se</th>
      <th>fractal_dimension_se</th>
      <th>radius_worst</th>
      <th>texture_worst</th>
      <th>perimeter_worst</th>
      <th>area_worst</th>
      <th>smoothness_worst</th>
      <th>compactness_worst</th>
      <th>concavity_worst</th>
      <th>concave points_worst</th>
      <th>symmetry_worst</th>
      <th>fractal_dimension_worst</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>842302</td>
      <td>M</td>
      <td>17.99</td>
      <td>10.38</td>
      <td>122.80</td>
      <td>1001.0</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.3001</td>
      <td>0.14710</td>
      <td>0.2419</td>
      <td>0.07871</td>
      <td>1.0950</td>
      <td>0.9053</td>
      <td>8.589</td>
      <td>153.40</td>
      <td>0.006399</td>
      <td>0.04904</td>
      <td>0.05373</td>
      <td>0.01587</td>
      <td>0.03003</td>
      <td>0.006193</td>
      <td>25.38</td>
      <td>17.33</td>
      <td>184.60</td>
      <td>2019.0</td>
      <td>0.1622</td>
      <td>0.6656</td>
      <td>0.7119</td>
      <td>0.2654</td>
      <td>0.4601</td>
      <td>0.11890</td>
    </tr>
    <tr>
      <th>1</th>
      <td>842517</td>
      <td>M</td>
      <td>20.57</td>
      <td>17.77</td>
      <td>132.90</td>
      <td>1326.0</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.0869</td>
      <td>0.07017</td>
      <td>0.1812</td>
      <td>0.05667</td>
      <td>0.5435</td>
      <td>0.7339</td>
      <td>3.398</td>
      <td>74.08</td>
      <td>0.005225</td>
      <td>0.01308</td>
      <td>0.01860</td>
      <td>0.01340</td>
      <td>0.01389</td>
      <td>0.003532</td>
      <td>24.99</td>
      <td>23.41</td>
      <td>158.80</td>
      <td>1956.0</td>
      <td>0.1238</td>
      <td>0.1866</td>
      <td>0.2416</td>
      <td>0.1860</td>
      <td>0.2750</td>
      <td>0.08902</td>
    </tr>
    <tr>
      <th>2</th>
      <td>84300903</td>
      <td>M</td>
      <td>19.69</td>
      <td>21.25</td>
      <td>130.00</td>
      <td>1203.0</td>
      <td>0.10960</td>
      <td>0.15990</td>
      <td>0.1974</td>
      <td>0.12790</td>
      <td>0.2069</td>
      <td>0.05999</td>
      <td>0.7456</td>
      <td>0.7869</td>
      <td>4.585</td>
      <td>94.03</td>
      <td>0.006150</td>
      <td>0.04006</td>
      <td>0.03832</td>
      <td>0.02058</td>
      <td>0.02250</td>
      <td>0.004571</td>
      <td>23.57</td>
      <td>25.53</td>
      <td>152.50</td>
      <td>1709.0</td>
      <td>0.1444</td>
      <td>0.4245</td>
      <td>0.4504</td>
      <td>0.2430</td>
      <td>0.3613</td>
      <td>0.08758</td>
    </tr>
    <tr>
      <th>3</th>
      <td>84348301</td>
      <td>M</td>
      <td>11.42</td>
      <td>20.38</td>
      <td>77.58</td>
      <td>386.1</td>
      <td>0.14250</td>
      <td>0.28390</td>
      <td>0.2414</td>
      <td>0.10520</td>
      <td>0.2597</td>
      <td>0.09744</td>
      <td>0.4956</td>
      <td>1.1560</td>
      <td>3.445</td>
      <td>27.23</td>
      <td>0.009110</td>
      <td>0.07458</td>
      <td>0.05661</td>
      <td>0.01867</td>
      <td>0.05963</td>
      <td>0.009208</td>
      <td>14.91</td>
      <td>26.50</td>
      <td>98.87</td>
      <td>567.7</td>
      <td>0.2098</td>
      <td>0.8663</td>
      <td>0.6869</td>
      <td>0.2575</td>
      <td>0.6638</td>
      <td>0.17300</td>
    </tr>
    <tr>
      <th>4</th>
      <td>84358402</td>
      <td>M</td>
      <td>20.29</td>
      <td>14.34</td>
      <td>135.10</td>
      <td>1297.0</td>
      <td>0.10030</td>
      <td>0.13280</td>
      <td>0.1980</td>
      <td>0.10430</td>
      <td>0.1809</td>
      <td>0.05883</td>
      <td>0.7572</td>
      <td>0.7813</td>
      <td>5.438</td>
      <td>94.44</td>
      <td>0.011490</td>
      <td>0.02461</td>
      <td>0.05688</td>
      <td>0.01885</td>
      <td>0.01756</td>
      <td>0.005115</td>
      <td>22.54</td>
      <td>16.67</td>
      <td>152.20</td>
      <td>1575.0</td>
      <td>0.1374</td>
      <td>0.2050</td>
      <td>0.4000</td>
      <td>0.1625</td>
      <td>0.2364</td>
      <td>0.07678</td>
    </tr>
  </tbody>
</table>
</div>




```python
all_df.columns
```




    Index(['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
           'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
           'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
           'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
           'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
           'fractal_dimension_se', 'radius_worst', 'texture_worst',
           'perimeter_worst', 'area_worst', 'smoothness_worst',
           'compactness_worst', 'concavity_worst', 'concave points_worst',
           'symmetry_worst', 'fractal_dimension_worst'],
          dtype='object')




```python
# Id column is redundant and not useful, we want to drop it
all_df.drop('id', axis =1, inplace=True)
```


```python
from sklearn.preprocessing import LabelEncoder

# transform the class labels from their original string representation (M and B) into integers
le = LabelEncoder()
all_df['diagnosis'] = le.fit_transform(all_df['diagnosis'])
```


```python
X = all_df.drop('diagnosis', axis=1) # drop labels for training set
y = all_df['diagnosis']
```


```python
# Divide records in training and testing sets: stratified sampling
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7, stratify=y)
```


```python
# Normalize the  data (center around 0 and scale to remove the variance).
scaler = StandardScaler()
Xs_train = scaler.fit_transform(X_train)
```

### 6.2 Evaluate Algorithms: Baseline


```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Spot-Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# Test options and evaluation metric
num_folds = 10
num_instances = len(X_train)
scoring = 'accuracy'
results = []
names = []
for name, model in models:
    kf = KFold(n_splits=num_folds, random_state=rnd_seed)
    cv_results = cross_val_score(model, X_train, y_train, cv=kf, scoring=scoring, n_jobs=-1)
    results.append(cv_results)
    names.append(name)
```


```python
print('10-Fold cross-validation accuracy score for the training data for all the classifiers') 
for name, cv_results in zip(names, results):
    print("%-10s: %.6f (%.6f)" % (name, cv_results.mean(), cv_results.std()))
```

    10-Fold cross-validation accuracy score for the training data for all the classifiers
    LR        : 0.952372 (0.041013)
    LDA       : 0.967308 (0.035678)
    KNN       : 0.932179 (0.037324)
    CART      : 0.969936 (0.029144)
    NB        : 0.937308 (0.042266)
    SVM       : 0.627885 (0.070174)
    

**Observation**

The results suggest That both Logistic Regression and LDA may be worth further study. These are just mean accuracy values. It is always wise to look at the distribution of accuracy values calculated across cross validation folds. We can do that graphically using box and whisker plots.


```python
# Compare Algorithms
plt.title( 'Algorithm Comparison' )
plt.boxplot(results)
plt.xlabel('Classifiers')
plt.ylabel('10 Fold CV Scores')
plt.xticks(np.arange(len(names)) + 1, names);
```


![png](output_154_0.png)


**Observation**

The results show a similar tight distribution for all classifiers except SVM which is encouraging, suggesting low variance. The poor results for SVM are surprising.

It is possible the varied distribution of the attributes may have an effect on the accuracy of algorithms such as SVM. In the next section we will repeat this spot-check with a standardized copy of the training dataset.

### 6.3 Evaluate Algorithms: Standardize Data


```python
# Standardize the dataset
pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR', LogisticRegression())])))
pipelines.append(('ScaledLDA', Pipeline([('Scaler', StandardScaler()),('LDA', LinearDiscriminantAnalysis())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsClassifier())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeClassifier())])))
pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('NB', GaussianNB())])))
pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()),('SVM', SVC())])))

results = []
names = []
for name, model in pipelines:
    kf = KFold(n_splits=num_folds, random_state=rnd_seed)
    cv_results = cross_val_score(model, X_train, y_train, cv=kf, scoring=scoring, n_jobs=-1)
    results.append(cv_results)
    names.append(name)
```


```python
print('10-Fold cross-validation accuracy score for the training data for all the classifiers') 
for name, cv_results in zip(names, results):
    print("%-10s: %.6f (%.6f)" % (name, cv_results.mean(), cv_results.std()))
```

    10-Fold cross-validation accuracy score for the training data for all the classifiers
    ScaledLR  : 0.984936 (0.022942)
    ScaledLDA : 0.967308 (0.035678)
    ScaledKNN : 0.952179 (0.038156)
    ScaledCART: 0.934679 (0.034109)
    ScaledNB  : 0.937244 (0.043887)
    ScaledSVM : 0.969936 (0.038398)
    


```python
# Compare Algorithms
plt.title( 'Algorithm Comparison' )
plt.boxplot(results)
plt.xlabel('Classifiers')
plt.ylabel('10 Fold CV Scores')
plt.xticks(np.arange(len(names)) + 1, names, rotation="90");
```


![png](output_159_0.png)


**Observations**

The results show that standardization of the data has lifted the skill of SVM to be the most accurate algorithm tested so far.

The results suggest digging deeper into the SVM and LDA and LR algorithms. It is very likely that configuration beyond the default may yield even more accurate models.

### 6.4 Algorithm Tuning
In this section we investigate tuning the parameters for three algorithms that show promise from the spot-checking in the previous section: LR, LDA and SVM.

#### Tuning hyper-parameters - SVC estimator


```python
# Make Support Vector Classifier Pipeline
pipe_svc = Pipeline([('scl', StandardScaler()),
                     ('pca', PCA(n_components=2)),
                     ('clf', SVC(probability=True, verbose=False))])
```


```python
# Fit Pipeline to training data and score
scores = cross_val_score(estimator=pipe_svc, X=X_train, y=y_train, cv=10, n_jobs=-1, verbose=0)
print('SVC Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))
```

    SVC Model Training Accuracy: 0.942 +/- 0.034
    


```python
from sklearn.model_selection import GridSearchCV

# Tune Hyperparameters
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'clf__C': param_range,'clf__kernel': ['linear']},
              {'clf__C': param_range,'clf__gamma': param_range, 'clf__kernel': ['rbf']}]

gs_svc = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=10,
                  n_jobs=-1)

gs_svc = gs_svc.fit(X_train, y_train)
```


```python
gs_svc.best_estimator_.named_steps
```




    {'clf': SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
       decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
       max_iter=-1, probability=True, random_state=None, shrinking=True,
       tol=0.001, verbose=False),
     'pca': PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,
       svd_solver='auto', tol=0.0, whiten=False),
     'scl': StandardScaler(copy=True, with_mean=True, with_std=True)}




```python
gs_svc.best_estimator_.named_steps['clf'].coef_
```




    array([[ 1.57606226, -0.87384284]])




```python
gs_svc.best_estimator_.named_steps['clf'].support_vectors_
```




    array([[ -5.59298401e-03,   2.54545060e-02],
           [ -3.36380410e-01,  -2.57254998e-01],
           [ -3.38622032e-01,  -7.19898441e-01],
           [ -7.04681309e-01,  -2.09847293e+00],
           [ -1.29967755e+00,  -1.62913054e+00],
           [ -8.48983391e-02,  -1.45496113e-01],
           [ -4.64780833e-01,  -9.01859111e-01],
           [  1.42724855e+00,   1.42660623e+00],
           [ -7.60785538e-01,  -1.16034158e+00],
           [  2.88483593e+00,   4.20900482e+00],
           [  1.94950775e+00,   2.36149488e+00],
           [ -1.54668166e+00,  -4.47823571e+00],
           [ -1.05181400e+00,  -1.30862774e+00],
           [  6.53277729e+00,   1.24974670e+01],
           [ -1.18800512e+00,  -1.55908705e+00],
           [ -6.16694586e-01,  -1.43967224e+00],
           [ -6.72611104e-01,  -1.22372306e+00],
           [  2.19235999e+00,   4.45143040e+00],
           [  1.27634550e+00,   1.13317453e+00],
           [ -4.60409592e-01,  -2.02632100e-01],
           [  5.54733653e-02,  -4.71520085e-02],
           [  1.33960706e+00,   2.17971509e+00],
           [  3.26676149e-01,   1.04285573e+00],
           [  1.89591695e-01,  -3.93198289e-01],
           [ -7.26372775e-01,  -3.06086751e+00],
           [ -2.78661492e-01,  -8.85635475e-01],
           [ -8.90826277e-01,  -2.18409521e+00],
           [  2.78146485e+00,   3.54832149e+00],
           [  1.34343228e+00,   9.68287874e-01],
           [ -1.79989870e+00,  -3.06802592e+00],
           [  6.31320317e-01,   6.53514981e-01],
           [  3.13050289e-01,  -4.50638339e-01],
           [  5.24004417e-01,   4.90054487e-01],
           [  2.38717629e+00,   4.88835134e+00],
           [ -5.66948440e-01,  -2.04500537e+00],
           [ -1.72281144e-01,  -3.97083911e-02],
           [  1.76756731e+00,   2.44765347e+00],
           [  2.14777940e+00,   2.37940489e+00],
           [  2.41815845e+00,   4.03922716e+00],
           [  7.60056497e-01,   5.17796680e-01],
           [ -2.38441481e+00,  -8.85474067e-01],
           [  8.59240050e-01,   1.01088149e+00],
           [ -1.13631837e-01,  -5.81038254e-01],
           [ -2.70785812e-01,   2.35457460e-01],
           [ -6.27711304e-01,  -2.34696985e+00],
           [ -8.73772942e-01,  -1.66619665e+00],
           [ -7.25279424e-01,  -2.64156929e+00],
           [ -3.71246204e-01,  -1.39306856e+00],
           [  5.61655769e-01,   3.87421293e-01],
           [  1.74473473e+00,   1.57197298e+00]])




```python
print('SVC Model Tuned Parameters Best Score: ', gs_svc.best_score_)
print('SVC Model Best Parameters: ', gs_svc.best_params_)
```

    SVC Model Tuned Parameters Best Score:  0.957286432161
    SVC Model Best Parameters:  {'clf__C': 1.0, 'clf__kernel': 'linear'}
    

#### Tuning the hyper-parameters - k-NN hyperparameters
 For our standard k-NN implementation, there are two primary hyperparameters that we’ll want to tune:

* The number of neighbors k.
* The distance metric/similarity function.

Both of these values can dramatically affect the accuracy of our k-NN classifier. Grid object is ready to do 10-fold cross validation on a KNN model using classification accuracy as the evaluation metric. In addition, there is a parameter grid to repeat the 10-fold cross validation process 30 times. Each time, the n_neighbors parameter should be given a different value from the list.

We can't give `GridSearchCV` just a list  
We've to specify `n_neighbors` should take on 1 through 30  
We can set `n_jobs` = -1 to run computations in parallel (if supported by your computer and OS) 


```python
from sklearn.neighbors import KNeighborsClassifier as KNN

pipe_knn = Pipeline([('scl', StandardScaler()),
                     ('pca', PCA(n_components=2)),
                     ('clf', KNeighborsClassifier())])
```


```python
#Fit Pipeline to training data and score
scores = cross_val_score(estimator=pipe_knn, 
                         X=X_train, 
                         y=y_train, 
                         cv=10,
                         n_jobs=-1)
print('Knn Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))
```

    Knn Model Training Accuracy: 0.945 +/- 0.027
    


```python
# Tune Hyperparameters
param_range = range(1, 31)
param_grid = [{'clf__n_neighbors': param_range}]
# instantiate the grid
grid = GridSearchCV(estimator=pipe_knn, 
                    param_grid=param_grid, 
                    cv=10, 
                    scoring='accuracy',
                    n_jobs=-1)
gs_knn = grid.fit(X_train, y_train)
```


```python
print('Knn Model Tuned Parameters Best Score: ', gs_knn.best_score_)
print('Knn Model Best Parameters: ', gs_knn.best_params_)
```

    Knn Model Tuned Parameters Best Score:  0.947236180905
    Knn Model Best Parameters:  {'clf__n_neighbors': 6}
    

### 6.5 Finalize Model


```python
# Use best parameters
final_clf_svc = gs_svc.best_estimator_

# Get Final Scores
scores = cross_val_score(estimator=final_clf_svc,
                         X=X_train,
                         y=y_train,
                         cv=10,
                         n_jobs=-1)
```


```python
print('Final Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))
print('Final Accuracy on Test set: %.5f' % final_clf_svc.score(X_test, y_test))
```

    Final Model Training Accuracy: 0.957 +/- 0.027
    Final Accuracy on Test set: 0.95322
    


```python
#clf_svc.fit(X_train, y_train)
y_pred = final_clf_svc.predict(X_test)
```


```python
print(accuracy_score(y_test, y_pred))
```

    0.953216374269
    


```python
print(confusion_matrix(y_test, y_pred))
```

    [[105   2]
     [  6  58]]
    


```python
print(classification_report(y_test, y_pred))
```

                 precision    recall  f1-score   support
    
              0       0.95      0.98      0.96       107
              1       0.97      0.91      0.94        64
    
    avg / total       0.95      0.95      0.95       171
    
    


```python

```

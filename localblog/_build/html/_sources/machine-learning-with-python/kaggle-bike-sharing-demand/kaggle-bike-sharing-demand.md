## Kaggle - Predicting Bike Sharing Demand

**Problem Statement** 

Bike sharing systems are a means of renting bicycles where the process of obtaining membership, rental, and bike return is automated via a network of kiosk locations throughout a city. Using these Bike Sharing systems, people rent a bike from one location and return it to a different or same place on need basis. People can rent a bike through membership (mostly regular users) or on demand basis (mostly casual users). This process is controlled by a network of automated kiosk across the city.

![bike sharing](bikes.png)

In [Kaggle Bike Sharing Demand](https://www.kaggle.com/c/bike-sharing-demand), the participants were asked to forecast bike rental demand of Bike sharing program in Washington, D.C. based on historical usage patterns in relation with weather, time and other data.

**Evaluation** 

Submissions are evaluated one the Root Mean Squared Logarithmic Error (RMSLE). The RMSLE is calculated as

$$\sqrt{\frac{1}{n}\sum_{i=1}^n (log(p_i + 1) - log(a_i + 1))^2}$$
where:
+ $n$ is the number of hours in the test set 
+ $p_i$ is your predicted count
+ $a_i$ is the actual count
+ log(x) is the natural logarithm

I have compiled this notebook by collating theory and codes from other blogs as well along with my own implementations. Wherever I have copied the theory/codes verbatim I have highlighted the references with <sup>[1]</sup> to give the authors their due credit. Please refer to the reference section at the end of the notebook for the original blogs of the respective authors.


```python
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import seaborn as sns
import matplotlib.pyplot as plt
```


```python
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 30)

#sns.set_style("whitegrid")
#plt.style.use('bmh')
plt.style.use('seaborn-whitegrid')

# this allows plots to appear directly in the notebook
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
```

### 1. Hypothesis Generation

Before exploring the data to understand the relationship between variables, it is recommended that we focus on hypothesis generation first. Now, this might sound counter-intuitive for solving a data science problem, but before exploring data, we should spend some time thinking about the business problem, gaining the domain knowledge and may be gaining first hand experience of the problem.

How does it help? This practice usually helps us form better features later on, which are not biased by the data available in the dataset. At this stage, we are expected to posses structured thinking i.e. a thinking process which takes into consideration all the possible aspects of a particular problem.

Here are some of the hypothesis which could influence the demand of bikes:

**Hourly trend:** There must be high demand during office timings. Early morning and late evening can have different trend (cyclist) and low demand during 10:00 pm to 4:00 am.

**Daily Trend:** Registered users demand more bike on weekdays as compared to weekend or holiday.

**Rain:** The demand of bikes will be lower on a rainy day as compared to a sunny day. Similarly, higher humidity will cause to lower the demand and vice versa.

**Temperature:** In India, temperature has negative correlation with bike demand. But, after looking at Washington's temperature graph, we presume it may have positive correlation.

**Pollution:** If the pollution level in a city starts soaring, people may start using Bike (it may be influenced by government / company policies or increased awareness).

**Time:** Total demand should have higher contribution of registered user as compared to casual because registered user base would increase over time.

**Traffic:** It can be positively correlated with Bike demand. Higher traffic may force people to use bike as compared to other road transport medium like car, taxi etc.

### 2. Understanding the Data Set

The dataset shows hourly rental data for two years (2011 and 2012). The training data set is for the first 19 days of each month. The test dataset is from 20th day to month's end. We are required to predict the total count of bikes rented during each hour covered by the test set.

In the training data set, they have separately given bike demand by registered, casual users and sum of both is given as count.

Training data set has 12 variables (see below) and Test has 9 (excluding registered, casual and count).

<p style="text-align: justify;"><strong>Independent Variables</strong></p>
<pre><strong>datetime:   </strong>date and hour in "mm/dd/yyyy hh:mm" format
<strong>season:</strong> &nbsp;   Four categories-&gt; 1 = spring, 2 = summer, 3 = fall, 4 = winter
<strong>holiday:</strong>    whether the day is a holiday or not (1/0)
<strong>workingday:</strong> whether the day is neither a weekend nor holiday (1/0)
<strong>weather:</strong>&nbsp;   Four Categories of weather
            1-&gt; Clear, Few clouds, Partly cloudy
            2-&gt; Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
            3-&gt; Light Snow and Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
            4-&gt; Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
<strong>temp:</strong>       hourly temperature in Celsius
<strong>atemp:</strong>      "feels like" temperature in Celsius
<strong>humidity:</strong>   relative humidity
<strong>windspeed:</strong>  wind speed
</pre>

<p><strong>Dependent Variables</strong></p>
<pre><strong>registered:</strong> number of registered user
<strong>casual:</strong>     number of non-registered user
<strong>count:</strong>      number of total rentals (registered + casual)
</pre> 

### 3. Importing Data Set and Basic Data Exploration


```python
train_df = pd.read_csv('data/train.csv')
train_df['data_set'] = 'train'
train_df.head(5)
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
      <th>datetime</th>
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>casual</th>
      <th>registered</th>
      <th>count</th>
      <th>data_set</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-01-01 00:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>81</td>
      <td>0.0</td>
      <td>3</td>
      <td>13</td>
      <td>16</td>
      <td>train</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-01-01 01:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0</td>
      <td>8</td>
      <td>32</td>
      <td>40</td>
      <td>train</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-01-01 02:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0</td>
      <td>5</td>
      <td>27</td>
      <td>32</td>
      <td>train</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-01-01 03:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75</td>
      <td>0.0</td>
      <td>3</td>
      <td>10</td>
      <td>13</td>
      <td>train</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-01-01 04:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>train</td>
    </tr>
  </tbody>
</table>
</div>


```python
test_df = pd.read_csv('data/test.csv')
test_df['data_set'] = 'test'
test_df.head(5)
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
      <th>datetime</th>
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>data_set</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-01-20 00:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>11.365</td>
      <td>56</td>
      <td>26.0027</td>
      <td>test</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-01-20 01:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>13.635</td>
      <td>56</td>
      <td>0.0000</td>
      <td>test</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-01-20 02:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>13.635</td>
      <td>56</td>
      <td>0.0000</td>
      <td>test</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-01-20 03:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>12.880</td>
      <td>56</td>
      <td>11.0014</td>
      <td>test</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-01-20 04:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>12.880</td>
      <td>56</td>
      <td>11.0014</td>
      <td>test</td>
    </tr>
  </tbody>
</table>
</div>



#### 3.1. Combine both Train and Test Data set (to understand the distribution of independent variable together).


```python
# combine train and test data into one df
test_df['registered'] = 0
test_df['casual'] = 0
test_df['count'] = 0

all_df = pd.concat([train_df, test_df])
all_df.head(5)
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
      <th>atemp</th>
      <th>casual</th>
      <th>count</th>
      <th>data_set</th>
      <th>datetime</th>
      <th>holiday</th>
      <th>humidity</th>
      <th>registered</th>
      <th>season</th>
      <th>temp</th>
      <th>weather</th>
      <th>windspeed</th>
      <th>workingday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14.395</td>
      <td>3</td>
      <td>16</td>
      <td>train</td>
      <td>2011-01-01 00:00:00</td>
      <td>0</td>
      <td>81</td>
      <td>13</td>
      <td>1</td>
      <td>9.84</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13.635</td>
      <td>8</td>
      <td>40</td>
      <td>train</td>
      <td>2011-01-01 01:00:00</td>
      <td>0</td>
      <td>80</td>
      <td>32</td>
      <td>1</td>
      <td>9.02</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13.635</td>
      <td>5</td>
      <td>32</td>
      <td>train</td>
      <td>2011-01-01 02:00:00</td>
      <td>0</td>
      <td>80</td>
      <td>27</td>
      <td>1</td>
      <td>9.02</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14.395</td>
      <td>3</td>
      <td>13</td>
      <td>train</td>
      <td>2011-01-01 03:00:00</td>
      <td>0</td>
      <td>75</td>
      <td>10</td>
      <td>1</td>
      <td>9.84</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14.395</td>
      <td>0</td>
      <td>1</td>
      <td>train</td>
      <td>2011-01-01 04:00:00</td>
      <td>0</td>
      <td>75</td>
      <td>1</td>
      <td>1</td>
      <td>9.84</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
all_df.tail(5)
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
      <th>atemp</th>
      <th>casual</th>
      <th>count</th>
      <th>data_set</th>
      <th>datetime</th>
      <th>holiday</th>
      <th>humidity</th>
      <th>registered</th>
      <th>season</th>
      <th>temp</th>
      <th>weather</th>
      <th>windspeed</th>
      <th>workingday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6488</th>
      <td>12.880</td>
      <td>0</td>
      <td>0</td>
      <td>test</td>
      <td>2012-12-31 19:00:00</td>
      <td>0</td>
      <td>60</td>
      <td>0</td>
      <td>1</td>
      <td>10.66</td>
      <td>2</td>
      <td>11.0014</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6489</th>
      <td>12.880</td>
      <td>0</td>
      <td>0</td>
      <td>test</td>
      <td>2012-12-31 20:00:00</td>
      <td>0</td>
      <td>60</td>
      <td>0</td>
      <td>1</td>
      <td>10.66</td>
      <td>2</td>
      <td>11.0014</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6490</th>
      <td>12.880</td>
      <td>0</td>
      <td>0</td>
      <td>test</td>
      <td>2012-12-31 21:00:00</td>
      <td>0</td>
      <td>60</td>
      <td>0</td>
      <td>1</td>
      <td>10.66</td>
      <td>1</td>
      <td>11.0014</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6491</th>
      <td>13.635</td>
      <td>0</td>
      <td>0</td>
      <td>test</td>
      <td>2012-12-31 22:00:00</td>
      <td>0</td>
      <td>56</td>
      <td>0</td>
      <td>1</td>
      <td>10.66</td>
      <td>1</td>
      <td>8.9981</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6492</th>
      <td>13.635</td>
      <td>0</td>
      <td>0</td>
      <td>test</td>
      <td>2012-12-31 23:00:00</td>
      <td>0</td>
      <td>65</td>
      <td>0</td>
      <td>1</td>
      <td>10.66</td>
      <td>1</td>
      <td>8.9981</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# lowercase column names
all_df.columns = map(str.lower, all_df.columns)
all_df.columns
```




    Index(['atemp', 'casual', 'count', 'data_set', 'datetime', 'holiday',
           'humidity', 'registered', 'season', 'temp', 'weather', 'windspeed',
           'workingday'],
          dtype='object')




```python
# parse datetime colum & add new time related columns
dt = pd.DatetimeIndex(all_df['datetime'])
all_df.set_index(dt, inplace=True)
```


```python
all_df.head(5)
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
      <th>atemp</th>
      <th>casual</th>
      <th>count</th>
      <th>data_set</th>
      <th>datetime</th>
      <th>holiday</th>
      <th>humidity</th>
      <th>registered</th>
      <th>season</th>
      <th>temp</th>
      <th>weather</th>
      <th>windspeed</th>
      <th>workingday</th>
    </tr>
    <tr>
      <th>datetime</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2011-01-01 00:00:00</th>
      <td>14.395</td>
      <td>3</td>
      <td>16</td>
      <td>train</td>
      <td>2011-01-01 00:00:00</td>
      <td>0</td>
      <td>81</td>
      <td>13</td>
      <td>1</td>
      <td>9.84</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2011-01-01 01:00:00</th>
      <td>13.635</td>
      <td>8</td>
      <td>40</td>
      <td>train</td>
      <td>2011-01-01 01:00:00</td>
      <td>0</td>
      <td>80</td>
      <td>32</td>
      <td>1</td>
      <td>9.02</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2011-01-01 02:00:00</th>
      <td>13.635</td>
      <td>5</td>
      <td>32</td>
      <td>train</td>
      <td>2011-01-01 02:00:00</td>
      <td>0</td>
      <td>80</td>
      <td>27</td>
      <td>1</td>
      <td>9.02</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2011-01-01 03:00:00</th>
      <td>14.395</td>
      <td>3</td>
      <td>13</td>
      <td>train</td>
      <td>2011-01-01 03:00:00</td>
      <td>0</td>
      <td>75</td>
      <td>10</td>
      <td>1</td>
      <td>9.84</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2011-01-01 04:00:00</th>
      <td>14.395</td>
      <td>0</td>
      <td>1</td>
      <td>train</td>
      <td>2011-01-01 04:00:00</td>
      <td>0</td>
      <td>75</td>
      <td>1</td>
      <td>1</td>
      <td>9.84</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Indexing by datetime let's us to select rows by specifying time ranges
all_df['2011-01-01 01:00:00':'2011-01-01 03:00:00']
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
      <th>atemp</th>
      <th>casual</th>
      <th>count</th>
      <th>data_set</th>
      <th>datetime</th>
      <th>holiday</th>
      <th>humidity</th>
      <th>registered</th>
      <th>season</th>
      <th>temp</th>
      <th>weather</th>
      <th>windspeed</th>
      <th>workingday</th>
    </tr>
    <tr>
      <th>datetime</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2011-01-01 01:00:00</th>
      <td>13.635</td>
      <td>8</td>
      <td>40</td>
      <td>train</td>
      <td>2011-01-01 01:00:00</td>
      <td>0</td>
      <td>80</td>
      <td>32</td>
      <td>1</td>
      <td>9.02</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2011-01-01 02:00:00</th>
      <td>13.635</td>
      <td>5</td>
      <td>32</td>
      <td>train</td>
      <td>2011-01-01 02:00:00</td>
      <td>0</td>
      <td>80</td>
      <td>27</td>
      <td>1</td>
      <td>9.02</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2011-01-01 03:00:00</th>
      <td>14.395</td>
      <td>3</td>
      <td>13</td>
      <td>train</td>
      <td>2011-01-01 03:00:00</td>
      <td>0</td>
      <td>75</td>
      <td>10</td>
      <td>1</td>
      <td>9.84</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# find missing values in dataset if any
all_df.isnull().values.sum()
```




    0



#### 3.2. Understand the distribution of numerical variables and generate a frequency table for numeric variables.


```python
plt.figure(figsize=(20,15))
plt.subplot(421)
all_df['season'].plot.hist(bins=10, color='blue', label='Histogram of Season', edgecolor='black')
plt.legend(loc='best')
plt.subplot(422)
all_df['weather'].plot.hist(bins=10, color='green', label='Histogram of Weather', edgecolor='black')
plt.legend(loc='best')
plt.subplot(423)
all_df['humidity'].plot.hist(bins=10, color='orange', label='Histogram of Humidity', edgecolor='black')
plt.legend(loc='best')
plt.subplot(424)
all_df['holiday'].plot.hist(bins=10, color='pink', label='Histogram of Holiday', edgecolor='black')
plt.legend(loc='best')
plt.subplot(425)
all_df['workingday'].plot.hist(bins=10, color='red', label='Histogram of Working Day', edgecolor='black')
plt.legend(loc='best')
plt.subplot(426)
all_df['temp'].plot.hist(bins=10, color='yellow', label='Histogram of Temperature', edgecolor='black')
plt.legend(loc='best')
plt.subplot(427)
all_df['atemp'].plot.hist(bins=10, color='cyan', label='Histogram of Feels Like Temp', edgecolor='black')
plt.legend(loc='best')
plt.subplot(428)
all_df['windspeed'].plot.hist(bins=10, color='purple', label='Histogram of Windpseed', edgecolor='black')
plt.legend(loc='best')
plt.tight_layout();
```


![png](output_17_0.png)


Few inferences can be drawn by looking at the these histograms:
- Season has four categories of almost equal distribution
- Weather 1 has higher contribution i.e. mostly clear weather.
- As expected, mostly working days and variable holiday is also showing a similar inference. You can use the code above to look at the distribution in detail. Here you can generate a variable for weekday using holiday and working day. Incase, if both have zero values, then it must be a working day.
- Variables temp, atemp, humidity and windspeed  looks naturally distributed.


```python
# logarithmic transformation of dependent cols
# (adding 1 first so that 0 values don't become -inf)
for col in ['casual', 'registered', 'count']:
    all_df['%s_log' % col] = np.log(all_df[col] + 1)
```


```python
all_df['date'] = dt.date
all_df['day'] = dt.day
all_df['month'] = dt.month
all_df['year'] = dt.year
all_df['hour'] = dt.hour
all_df['dow'] = dt.dayofweek
all_df['woy'] = dt.weekofyear
```


```python
# How many columns have null values
all_df.isnull().sum()
```




    atemp             0
    casual            0
    count             0
    data_set          0
    datetime          0
    holiday           0
    humidity          0
    registered        0
    season            0
    temp              0
    weather           0
    windspeed         0
    workingday        0
    casual_log        0
    registered_log    0
    count_log         0
    date              0
    day               0
    month             0
    year              0
    hour              0
    dow               0
    woy               0
    dtype: int64




```python
# interpolate weather, temp, atemp, humidity, windspeed
all_df["weather"] = all_df["weather"].interpolate(method='time').apply(np.round)
all_df["temp"] = all_df["temp"].interpolate(method='time')
all_df["atemp"] = all_df["atemp"].interpolate(method='time')
all_df["humidity"] = all_df["humidity"].interpolate(method='time').apply(np.round)
all_df["windspeed"] = all_df["windspeed"].interpolate(method='time')
```


```python
# add a count_season column using join
by_season = all_df[all_df['data_set'] == 'train'].copy().groupby(['season'])[['count']].agg(sum)
by_season.columns = ['count_season']
all_df = all_df.join(by_season, on='season')
```


```python
print(by_season)
```

            count_season
    season              
    1             312498
    2             588282
    3             640662
    4             544034



```python
by_season.plot(kind='barh')
plt.grid(True)
plt.show();
```


![png](output_25_0.png)



```python
def get_day(day_start):
    day_end = day_start + pd.offsets.DateOffset(hours=23)
    return pd.date_range(day_start, day_end, freq="H")

# tax day
all_df.loc[get_day(pd.datetime(2011, 4, 15)), "workingday"] = 1
all_df.loc[get_day(pd.datetime(2012, 4, 16)), "workingday"] = 1

# thanksgiving friday
all_df.loc[get_day(pd.datetime(2011, 11, 25)), "workingday"] = 0
all_df.loc[get_day(pd.datetime(2012, 11, 23)), "workingday"] = 0

# tax day
all_df.loc[get_day(pd.datetime(2011, 4, 15)), "holiday"] = 0
all_df.loc[get_day(pd.datetime(2012, 4, 16)), "holiday"] = 0

# thanksgiving friday
all_df.loc[get_day(pd.datetime(2011, 11, 25)), "holiday"] = 1
all_df.loc[get_day(pd.datetime(2012, 11, 23)), "holiday"] = 1

#storms
all_df.loc[get_day(pd.datetime(2012, 5, 21)), "holiday"] = 1

#tornado
all_df.loc[get_day(pd.datetime(2012, 6, 1)), "holiday"] = 1
```


```python
by_hour = all_df[all_df['data_set'] == 'train'].copy().groupby(['hour', 'workingday'])['count'].agg('sum').unstack()
by_hour.head(10)
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
      <th>workingday</th>
      <th>0</th>
      <th>1</th>
    </tr>
    <tr>
      <th>hour</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>13633</td>
      <td>11455</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10384</td>
      <td>4988</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7654</td>
      <td>2605</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3666</td>
      <td>1425</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1230</td>
      <td>1602</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1280</td>
      <td>7655</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2719</td>
      <td>31979</td>
    </tr>
    <tr>
      <th>7</th>
      <td>6318</td>
      <td>90650</td>
    </tr>
    <tr>
      <th>8</th>
      <td>15380</td>
      <td>149680</td>
    </tr>
    <tr>
      <th>9</th>
      <td>25324</td>
      <td>75586</td>
    </tr>
  </tbody>
</table>
</div>



**Hourly trend:** There must be high demand during office timings. Early morning and late evening can have different trend (cyclist) and low demand during 10:00 pm to 4:00 am.


```python
# rentals by hour, split by working day (or not)
by_hour.plot(kind='bar', figsize=(15,5), width=0.8);
plt.grid(True)
plt.tight_layout();
```


![png](output_29_0.png)


### 4. Hypothesis Testing (using multivariate analysis)

Till now, we have got a fair understanding of the data set. Now, let's test the hypothesis which we had generated earlier.  Here we have added some additional hypothesis from the dataset. Let's test them one by one:

**Hourly trend**: We don't have the variable 'hour' as part of data provides but we extracted it using the datetime column.
Let's plot the hourly trend of count over hours and check if our hypothesis is correct or not. We will separate train and test data set from combined one.


```python
train_df = all_df[all_df['data_set'] == 'train'].copy()
```


```python
#train_df.boxplot(column='count', by='hour', figsize=(15,5))
#plt.ylabel('Count of Users')
#plt.title("Boxplot of Count grouped by hour")
#plt.suptitle("") # get rid of the pandas autogenerated title
```


```python
fig, ax = plt.subplots(figsize=(18, 5))
sns.boxplot(x=train_df['hour'], y=train_df['count'], ax=ax)
ax.set_ylabel('Count of Users')
ax.set_title("Boxplot of Count grouped by hour");
#plt.suptitle("") # get rid of the pandas autogenerated title
```


![png](output_33_0.png)


Above, we can see the trend of bike demand over hours. Quickly, we'll segregate the bike demand in three categories:

- High     : 7-9 and 17-19 hours
- Average  : 10-16 hours
- Low      : 0-6 and 20-24 hours
Here we have analyzed the distribution of total bike demand. 

Let's look at the distribution of registered and casual users separately.

#### Good Weather is most frequent in Fall<sup>[5]</sup>


```python
season_map = {1:'Spring', 2:'Summer', 3:'Fall', 4:'Winter'}
good_weather = all_df[all_df['weather'] == 1][['hour', 'season']].copy()
data = pd.DataFrame({'count' : good_weather.groupby(["hour","season"]).size()}).reset_index()
data['season'] = data['season'].map(lambda d : season_map[d])

fig, ax = plt.subplots(figsize=(18, 5))
sns.pointplot(x=data["hour"], y=data["count"], hue=data["season"], ax=ax)
ax.set(xlabel='Hour Of The Day', ylabel='Good Weather Count', title="Good Weather By Hour Of The Day Across Season");
```


![png](output_36_0.png)


#### Normal Weather happens most frequent in Spring<sup>[5]</sup>


```python
season_map = {1:'Spring', 2:'Summer', 3:'Fall', 4:'Winter'}
normal_weather = all_df[all_df['weather'] == 3][['hour', 'season']].copy()
data = pd.DataFrame({'count' : normal_weather.groupby(["hour","season"]).size()}).reset_index()
data['season'] = data['season'].map(lambda d : season_map[d])
```


```python
data.sample(n=5)
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
      <th>hour</th>
      <th>season</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>50</th>
      <td>12</td>
      <td>Fall</td>
      <td>12</td>
    </tr>
    <tr>
      <th>46</th>
      <td>11</td>
      <td>Fall</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>Winter</td>
      <td>11</td>
    </tr>
    <tr>
      <th>63</th>
      <td>15</td>
      <td>Winter</td>
      <td>19</td>
    </tr>
    <tr>
      <th>33</th>
      <td>8</td>
      <td>Summer</td>
      <td>18</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots(figsize=(18, 5))
sns.pointplot(x=data["hour"], y=data["count"], hue=data["season"], ax=ax)
ax.set(xlabel='Hour Of The Day', ylabel='Normal Weather Count', title="Normal Weather By Hour Of The Day Across Season");
```


![png](output_40_0.png)



```python
data = pd.pivot_table(data, values='count', columns='season', index='hour')
data.sample(5)
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
      <th>season</th>
      <th>Fall</th>
      <th>Spring</th>
      <th>Summer</th>
      <th>Winter</th>
    </tr>
    <tr>
      <th>hour</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>18</th>
      <td>14</td>
      <td>18</td>
      <td>19</td>
      <td>18</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8</td>
      <td>14</td>
      <td>19</td>
      <td>16</td>
    </tr>
    <tr>
      <th>15</th>
      <td>15</td>
      <td>16</td>
      <td>16</td>
      <td>19</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12</td>
      <td>12</td>
      <td>14</td>
      <td>17</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>11</td>
      <td>8</td>
      <td>17</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots(figsize=(12, 5))
data.plot.area(stacked=False, ax=ax)
ax.set(xlabel='Hour Of The Day', ylabel='Normal Weather Count', title="Normal Weather By Hour Of The Day Across Season");
```


![png](output_42_0.png)


#### Bad Weather happens most frequent in Summer & Winter<sup>[5]</sup>


```python
season_map = {1:'Spring', 2:'Summer', 3:'Fall', 4:'Winter'}
bad_weather = all_df[all_df['weather'] == 3][['hour', 'season']].copy()
data = pd.DataFrame({'count' : bad_weather.groupby(["hour","season"]).size()}).reset_index()
data['season'] = data['season'].map(lambda d : season_map[d])

fig, ax = plt.subplots(figsize=(18, 5))
sns.pointplot(x=data["hour"], y=data["count"], hue=data["season"], ax=ax)
ax.set(xlabel='Hour Of The Day', ylabel='Bad Weather Count', title="Bad Weather By Hour Of The Day Across Season");
```


![png](output_44_0.png)


#### Bikes are rented more in Good Weather and much less in Bad Weather


```python
weather_map = {1:'Good', 2:'Normal', 3:'Bad', 4:'Worse'}
data = pd.DataFrame(train_df.groupby(["hour","weather"], sort=True)["count"].mean()).reset_index()
data['weather'] = data['weather'].map(lambda d : weather_map[d])
fig, ax = plt.subplots(figsize=(18, 5))
sns.pointplot(x=data["hour"], y=data["count"], hue=data["weather"], ax=ax)
ax.set(xlabel='Hour Of The Day', ylabel='Users Count', title="Average Users Count By Hour Of The Day Across Weather");
```


![png](output_46_0.png)


#### Bikes are rented more in Fall and much less in Spring


```python
season_map = {1:'Spring', 2:'Summer', 3:'Fall', 4:'Winter'}
data = pd.DataFrame({'mean':train_df.groupby(["hour","season"], sort=True)["count"].mean()}).reset_index()
data['season'] = data['season'].map(lambda d : season_map[d])
```


```python
data.sample(n=5)
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
      <th>hour</th>
      <th>season</th>
      <th>mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>48</th>
      <td>12</td>
      <td>Spring</td>
      <td>154.412281</td>
    </tr>
    <tr>
      <th>92</th>
      <td>23</td>
      <td>Spring</td>
      <td>45.333333</td>
    </tr>
    <tr>
      <th>27</th>
      <td>6</td>
      <td>Winter</td>
      <td>82.254386</td>
    </tr>
    <tr>
      <th>89</th>
      <td>22</td>
      <td>Summer</td>
      <td>154.192982</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>Summer</td>
      <td>58.473684</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots(figsize=(18, 5))
sns.pointplot(x=data["hour"], y=data["mean"], hue=data["season"], ax=ax)
ax.set(xlabel='Hour Of The Day', ylabel='Users Count', title="Average Users Count By Hour Of The Day Across Season");
```


![png](output_50_0.png)


#### Bikes are rented mostly for Morning/Evening commutes on Weekdays, and mostly Daytime rides on Weekends


```python
day_map = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 6:'Sunday'}
hueOrder = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
data = pd.DataFrame({'mean':train_df.groupby(["hour","dow"], sort=True)["count"].mean()}).reset_index()
data['dow'] = data['dow'].map(lambda d : day_map[d])
```


```python
data.sample(n=5)
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
      <th>hour</th>
      <th>dow</th>
      <th>mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>103</th>
      <td>14</td>
      <td>Saturday</td>
      <td>398.409091</td>
    </tr>
    <tr>
      <th>134</th>
      <td>19</td>
      <td>Tuesday</td>
      <td>356.123077</td>
    </tr>
    <tr>
      <th>55</th>
      <td>7</td>
      <td>Sunday</td>
      <td>34.742424</td>
    </tr>
    <tr>
      <th>78</th>
      <td>11</td>
      <td>Tuesday</td>
      <td>145.609375</td>
    </tr>
    <tr>
      <th>67</th>
      <td>9</td>
      <td>Friday</td>
      <td>262.406250</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots(figsize=(18, 5))
sns.pointplot(x=data["hour"], y=data["mean"], hue=data["dow"], hue_order=hueOrder, ax=ax)
ax.set(xlabel='Hour Of The Day', ylabel='Users Count', title="Average Users Count By Hour Of The Day Across Weekdays");
```


![png](output_54_0.png)


#### Renting patterns of Bikes are significantly different between Registered and Casual Users


```python
#fig, axs = plt.subplots(1, 2, figsize=(15,5), sharex=False, sharey=False)

#train_df.boxplot(column='casual', by='hour', ax=axs[0])
#axs[0].set_ylabel('casual users')
#axs[0].set_title('')

#train_df.boxplot(column='registered', by='hour', ax=axs[1])
#axs[1].set_ylabel('registered users')
#axs[1].set_title('')
```


```python
fig, axs = plt.subplots(1, 2, figsize=(18,5), sharex=False, sharey=False)

sns.boxplot(x='hour', y='casual', data=train_df, ax=axs[0])
axs[0].set_ylabel('casual users')
axs[0].set_title('')

sns.boxplot(x='hour', y='registered', data=train_df, ax=axs[1])
axs[1].set_ylabel('registered users')
axs[1].set_title('');
```


![png](output_57_0.png)



```python
train_df[["hour","casual","registered"]].head(5)
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
      <th>hour</th>
      <th>casual</th>
      <th>registered</th>
    </tr>
    <tr>
      <th>datetime</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2011-01-01 00:00:00</th>
      <td>0</td>
      <td>3</td>
      <td>13</td>
    </tr>
    <tr>
      <th>2011-01-01 01:00:00</th>
      <td>1</td>
      <td>8</td>
      <td>32</td>
    </tr>
    <tr>
      <th>2011-01-01 02:00:00</th>
      <td>2</td>
      <td>5</td>
      <td>27</td>
    </tr>
    <tr>
      <th>2011-01-01 03:00:00</th>
      <td>3</td>
      <td>3</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2011-01-01 04:00:00</th>
      <td>4</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.melt(train_df[["hour","casual","registered"]], id_vars=['hour'], value_vars=['casual', 'registered'], var_name='usertype', value_name='count').head(5)
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
      <th>hour</th>
      <th>usertype</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>casual</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>casual</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>casual</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>casual</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>casual</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots(figsize=(18, 5))
train_df_melt = pd.melt(train_df[["hour","casual","registered"]], id_vars=['hour'], value_vars=['casual', 'registered'], var_name='usertype', value_name='count')
data = pd.DataFrame(train_df_melt.groupby(["hour", "usertype"], sort=True)["count"].mean()).reset_index()
sns.pointplot(x=data["hour"], y=data["count"], hue=data["usertype"], hue_order=["casual","registered"], ax=ax)
ax.set(xlabel='Hour Of The Day', ylabel='Users Count', title='Average Users Count By Hour Of The Day Across User Type');
```


![png](output_60_0.png)


Above we can see that registered users have similar trend as count. Whereas, casual users have different trend. Thus, we can say that 'hour' is significant variable and our hypothesis is 'true'.

We can notice that there are a lot of outliers while plotting the count of registered and casual users. These values are not generated due to error, so we consider them as natural outliers. They might be a result of groups of people taking up cycling (who are not registered). To treat such outliers, we will use logarithm transformation. Let's look at the similar plot after log transformation.


```python
train_df = train_df.assign(log_count = lambda df : np.log(train_df['count']))

fig, ax = plt.subplots(figsize=(18, 5))
sns.boxplot(x='hour', y='log_count', data=train_df, ax=ax)
ax.set(ylabel='log(count) of Users',title='Boxplot of Log of Count grouped by hour')

#plt.suptitle("") # get rid of the pandas autogenerated title
train_df.drop(['log_count'], axis = 1, inplace=True);
```


![png](output_62_0.png)


#### On Workdays most Bikes are rented on Warm Mornings and Evenings<sup>[2]</sup>

<sup>[4]</sup>When graphing a categorical variable vs. a continuous variable, it can be useful to create a scatter plot to visually examine distributions. Together with a box plot, it will allow us to see the distributions of our variables. Unfortunately, if our points occur close together, we will get a very uninformative smear. One way of making the scatter plot work is by adding jitter. With the jitter, a random amount is added or subtracted to each of the variables along the categorical axis. Where before, we may have had a categorical value vector that looked something like [1,2,2,2,1,3], post-jitter, they would look something like [1.05, 1.96, 2.05, 2, .97, 2.95]. Each value has had somewhere between [-0.05,0.05] added to it. This then means that when we plot our variables, we'll see a cloud of points that represent our distribution, rather than a long smear.


```python
def hour_jitter(h):
    #return h + ((np.random.randint(low=0, high=9, size=1)[0] - 4) / 10)
    return h + np.random.uniform(-0.4, 0.4)
```


```python
def hour_format(h):
    return "{:02d}:00 AM".format(h) if h <= 12 else "{:02d}:00 PM".format(h%12)
```


```python
# jitter plot
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# color_map = plt.get_cmap("jet")
color_map = mcolors.ListedColormap(list(["#5e4fa2", "#3288bd", "#66c2a5", "#abdda4", "#e6f598", "#fee08b", "#fdae61", "#f46d43", "#d53e4f", "#9e0142"]))
train_df['hour_jitter'] = train_df['hour'].map(hour_jitter)
train_df[train_df['workingday'] == 1].plot(kind="scatter", x='hour_jitter', y='count',
    figsize=(18,6),
    c='temp', cmap=color_map, colorbar=True,
    sharex=False)

hours = np.unique(train_df['hour'].values)
hour_labels = [hour_format(h) for h in hours]
plt.xticks(hours, hour_labels, rotation='vertical');
```


![png](output_66_0.png)



```python
train_df.drop('hour_jitter', axis=1, inplace=True);
```

**Daily Trend:** Like Hour, we will generate a variable for day from datetime variable and after that we'll plot it.


```python
dayOfWeek={0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 6:'Sunday'}
all_df['weekday'] = all_df['dow'].map(dayOfWeek)

fig, axs = plt.subplots(1, 2, figsize=(15,5), sharex=False, sharey=False)

sns.boxplot(x='weekday', y='registered', data=all_df, ax=axs[0])
axs[0].set_ylabel('registered users')
axs[0].set_title('')

sns.boxplot(x='weekday', y='casual', data=all_df, ax=axs[1])
axs[1].set_ylabel('casual users')
axs[1].set_title('')

all_df.drop('weekday', axis=1, inplace=True);
```


![png](output_69_0.png)


**Rain:** We don’t have the 'rain' variable with us but have 'weather' which is sufficient to test our hypothesis. As per variable description, weather 3 represents light rain and weather 4 represents heavy rain. Take a look at the plot:


```python
fig, axs = plt.subplots(1, 2, figsize=(15,5), sharex=False, sharey=False)

sns.boxplot(x='weather', y='registered', data=all_df, ax=axs[0])
axs[0].set_ylabel('registered users')
axs[0].set_title('')

sns.boxplot(x='weather', y='casual', data=all_df, ax=axs[1])
axs[1].set_ylabel('casual users')
axs[1].set_title('');
```


![png](output_71_0.png)


It is clearly satisfying our hypothesis.

**Temperature, Windspeed and Humidity:** These are continuous variables so we can look at the correlation factor to validate hypothesis.

#### Correlation Between Count And Features


```python
sub_df = train_df[['count', 'registered', 'casual', 'temp', 'atemp', 'humidity', 'windspeed', 'workingday', 'holiday']]
```


```python
sub_df.corr()
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
      <th>count</th>
      <th>registered</th>
      <th>casual</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>workingday</th>
      <th>holiday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.000000</td>
      <td>0.970948</td>
      <td>0.690414</td>
      <td>0.394454</td>
      <td>0.389784</td>
      <td>-0.317371</td>
      <td>0.101369</td>
      <td>0.011965</td>
      <td>-0.008049</td>
    </tr>
    <tr>
      <th>registered</th>
      <td>0.970948</td>
      <td>1.000000</td>
      <td>0.497250</td>
      <td>0.318571</td>
      <td>0.314635</td>
      <td>-0.265458</td>
      <td>0.091052</td>
      <td>0.120154</td>
      <td>-0.023038</td>
    </tr>
    <tr>
      <th>casual</th>
      <td>0.690414</td>
      <td>0.497250</td>
      <td>1.000000</td>
      <td>0.467097</td>
      <td>0.462067</td>
      <td>-0.348187</td>
      <td>0.092276</td>
      <td>-0.319864</td>
      <td>0.040464</td>
    </tr>
    <tr>
      <th>temp</th>
      <td>0.394454</td>
      <td>0.318571</td>
      <td>0.467097</td>
      <td>1.000000</td>
      <td>0.984948</td>
      <td>-0.064949</td>
      <td>-0.017852</td>
      <td>0.033174</td>
      <td>0.002969</td>
    </tr>
    <tr>
      <th>atemp</th>
      <td>0.389784</td>
      <td>0.314635</td>
      <td>0.462067</td>
      <td>0.984948</td>
      <td>1.000000</td>
      <td>-0.043536</td>
      <td>-0.057473</td>
      <td>0.027851</td>
      <td>-0.003455</td>
    </tr>
    <tr>
      <th>humidity</th>
      <td>-0.317371</td>
      <td>-0.265458</td>
      <td>-0.348187</td>
      <td>-0.064949</td>
      <td>-0.043536</td>
      <td>1.000000</td>
      <td>-0.318607</td>
      <td>-0.011039</td>
      <td>0.012114</td>
    </tr>
    <tr>
      <th>windspeed</th>
      <td>0.101369</td>
      <td>0.091052</td>
      <td>0.092276</td>
      <td>-0.017852</td>
      <td>-0.057473</td>
      <td>-0.318607</td>
      <td>1.000000</td>
      <td>0.018454</td>
      <td>-0.000585</td>
    </tr>
    <tr>
      <th>workingday</th>
      <td>0.011965</td>
      <td>0.120154</td>
      <td>-0.319864</td>
      <td>0.033174</td>
      <td>0.027851</td>
      <td>-0.011039</td>
      <td>0.018454</td>
      <td>1.000000</td>
      <td>-0.213189</td>
    </tr>
    <tr>
      <th>holiday</th>
      <td>-0.008049</td>
      <td>-0.023038</td>
      <td>0.040464</td>
      <td>0.002969</td>
      <td>-0.003455</td>
      <td>0.012114</td>
      <td>-0.000585</td>
      <td>-0.213189</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
corrMatt = sub_df.corr()
mask = np.zeros_like(corrMatt)
mask[np.triu_indices_from(mask)] = True
fig, ax = plt.subplots(figsize=(15, 6))
sns.heatmap(corrMatt, mask=mask, vmax=.8, square=False, annot=True, ax=ax, linewidths=1);
```


![png](output_76_0.png)


Here are a few inferences you can draw by looking at the above histograms:

- Variable temp is positively correlated with dependent variables (casual is more compare to registered)
- Variable atemp is highly correlated with temp.
- Windspeed has lower correlation as compared to temp and humidity

**Time:** Let's extract year of each observation from the datetime column and see the trend of bike demand over year.

#### Distribution of data between Train and Test set based on Season


```python
season_map = {1:'Spring', 2:'Summer', 3:'Fall', 4:'Winter'}
data = all_df[['data_set', 'season']].copy()
data['season'] = data['season'].map(lambda d : season_map[d])
```


```python
sns.countplot(x="data_set", hue="season", data=data);
```


![png](output_80_0.png)


#### Distribution of data between Train and Test set based on Weather


```python
weather_map = {1:'Good', 2:'Normal', 3:'Bad', 4:'Worse'}
data = all_df[['data_set', 'weather']].copy()
data['weather'] = data['weather'].map(lambda d : weather_map[d])
```


```python
sns.countplot(x="data_set", hue="weather", data=data);
```


![png](output_83_0.png)


#### Distribution of count of users Train and Test set based on Year


```python
plt.figure(figsize=(8, 5))
sns.boxplot(x='year', y='count', data=train_df)
plt.ylabel('Count of Users')
plt.title("Boxplot of Count grouped by year");
```


![png](output_85_0.png)


We can see that 2012 has higher bike demand as compared to 2011.

**Pollution & Traffic:** We don't have the variable related with these metrics in our data set so we cannot test this hypothesis.

### 5. Feature Engineering

In addition to existing independent variables, we will create new variables to improve the prediction power of model. Initially, we have generated new variables like hour, month, day and year.


```python
# feature engineer a new column whether its a peak hour or not
all_df['peak'] = all_df[['hour', 'workingday']]\
    .apply(lambda df: 1 if ((df['workingday'] == 1 and (df['hour'] == 8 or 17 <= df['hour'] <= 18)) \
                            or (df['workingday'] == 0 and 10 <= df['workingday'] <= 19)) else 0, axis = 1)
```


```python
# sandy
all_df['holiday'] = all_df[['month', 'day', 'holiday', 'year']]\
    .apply(lambda df: 1 if (df['year'] == 2012 and df['month'] == 10 and df['day'] == 30) else 0, axis = 1)

# christmas and others
all_df['holiday'] = all_df[['month', 'day', 'holiday']]\
    .apply(lambda df: 1 if (df['month'] == 12 and df['day'] in [24, 26, 31]) else df['holiday'], axis = 1)
all_df['workingday'] = all_df[['month', 'day', 'workingday']]\
    .apply(lambda df: 0 if df['month'] == 12 and df['day'] in [24, 31] else df['workingday'], axis = 1)
```


```python
# from histogram
all_df['ideal'] = all_df[['temp', 'windspeed']]\
    .apply(lambda df: 1 if (df['temp'] > 27 and df['windspeed'] < 30) else 0, axis = 1)
    
all_df['sticky'] = all_df[['humidity', 'workingday']]\
    .apply(lambda df: 1 if (df['workingday'] == 1 and df['humidity'] >= 60) else 0, axis = 1)
```


```python
all_df.sample(10)
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
      <th>atemp</th>
      <th>casual</th>
      <th>count</th>
      <th>data_set</th>
      <th>datetime</th>
      <th>holiday</th>
      <th>humidity</th>
      <th>registered</th>
      <th>season</th>
      <th>temp</th>
      <th>weather</th>
      <th>windspeed</th>
      <th>workingday</th>
      <th>casual_log</th>
      <th>registered_log</th>
      <th>count_log</th>
      <th>date</th>
      <th>day</th>
      <th>month</th>
      <th>year</th>
      <th>hour</th>
      <th>dow</th>
      <th>woy</th>
      <th>count_season</th>
      <th>peak</th>
      <th>ideal</th>
      <th>sticky</th>
    </tr>
    <tr>
      <th>datetime</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2012-08-30 02:00:00</th>
      <td>29.545</td>
      <td>0</td>
      <td>0</td>
      <td>test</td>
      <td>2012-08-30 02:00:00</td>
      <td>0</td>
      <td>73</td>
      <td>0</td>
      <td>3</td>
      <td>25.42</td>
      <td>1</td>
      <td>0.0000</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2012-08-30</td>
      <td>30</td>
      <td>8</td>
      <td>2012</td>
      <td>2</td>
      <td>3</td>
      <td>35</td>
      <td>640662</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2012-08-14 20:00:00</th>
      <td>35.605</td>
      <td>65</td>
      <td>436</td>
      <td>train</td>
      <td>2012-08-14 20:00:00</td>
      <td>0</td>
      <td>74</td>
      <td>371</td>
      <td>3</td>
      <td>30.34</td>
      <td>1</td>
      <td>15.0013</td>
      <td>1</td>
      <td>4.189655</td>
      <td>5.918894</td>
      <td>6.079933</td>
      <td>2012-08-14</td>
      <td>14</td>
      <td>8</td>
      <td>2012</td>
      <td>20</td>
      <td>1</td>
      <td>33</td>
      <td>640662</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2012-05-01 04:00:00</th>
      <td>25.000</td>
      <td>1</td>
      <td>8</td>
      <td>train</td>
      <td>2012-05-01 04:00:00</td>
      <td>0</td>
      <td>72</td>
      <td>7</td>
      <td>2</td>
      <td>21.32</td>
      <td>2</td>
      <td>6.0032</td>
      <td>1</td>
      <td>0.693147</td>
      <td>2.079442</td>
      <td>2.197225</td>
      <td>2012-05-01</td>
      <td>1</td>
      <td>5</td>
      <td>2012</td>
      <td>4</td>
      <td>1</td>
      <td>18</td>
      <td>588282</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2011-07-13 19:00:00</th>
      <td>33.335</td>
      <td>79</td>
      <td>419</td>
      <td>train</td>
      <td>2011-07-13 19:00:00</td>
      <td>0</td>
      <td>79</td>
      <td>340</td>
      <td>3</td>
      <td>28.70</td>
      <td>1</td>
      <td>7.0015</td>
      <td>1</td>
      <td>4.382027</td>
      <td>5.831882</td>
      <td>6.040255</td>
      <td>2011-07-13</td>
      <td>13</td>
      <td>7</td>
      <td>2011</td>
      <td>19</td>
      <td>2</td>
      <td>28</td>
      <td>640662</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2012-02-05 09:00:00</th>
      <td>11.365</td>
      <td>9</td>
      <td>92</td>
      <td>train</td>
      <td>2012-02-05 09:00:00</td>
      <td>0</td>
      <td>70</td>
      <td>83</td>
      <td>1</td>
      <td>9.84</td>
      <td>2</td>
      <td>15.0013</td>
      <td>0</td>
      <td>2.302585</td>
      <td>4.430817</td>
      <td>4.532599</td>
      <td>2012-02-05</td>
      <td>5</td>
      <td>2</td>
      <td>2012</td>
      <td>9</td>
      <td>6</td>
      <td>5</td>
      <td>312498</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2011-05-22 02:00:00</th>
      <td>25.000</td>
      <td>0</td>
      <td>0</td>
      <td>test</td>
      <td>2011-05-22 02:00:00</td>
      <td>0</td>
      <td>94</td>
      <td>0</td>
      <td>2</td>
      <td>21.32</td>
      <td>1</td>
      <td>7.0015</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2011-05-22</td>
      <td>22</td>
      <td>5</td>
      <td>2011</td>
      <td>2</td>
      <td>6</td>
      <td>20</td>
      <td>588282</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2011-11-11 22:00:00</th>
      <td>15.910</td>
      <td>13</td>
      <td>91</td>
      <td>train</td>
      <td>2011-11-11 22:00:00</td>
      <td>0</td>
      <td>39</td>
      <td>78</td>
      <td>4</td>
      <td>13.12</td>
      <td>1</td>
      <td>12.9980</td>
      <td>0</td>
      <td>2.639057</td>
      <td>4.369448</td>
      <td>4.521789</td>
      <td>2011-11-11</td>
      <td>11</td>
      <td>11</td>
      <td>2011</td>
      <td>22</td>
      <td>4</td>
      <td>45</td>
      <td>544034</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2011-04-10 23:00:00</th>
      <td>22.725</td>
      <td>5</td>
      <td>32</td>
      <td>train</td>
      <td>2011-04-10 23:00:00</td>
      <td>0</td>
      <td>88</td>
      <td>27</td>
      <td>2</td>
      <td>18.86</td>
      <td>1</td>
      <td>23.9994</td>
      <td>0</td>
      <td>1.791759</td>
      <td>3.332205</td>
      <td>3.496508</td>
      <td>2011-04-10</td>
      <td>10</td>
      <td>4</td>
      <td>2011</td>
      <td>23</td>
      <td>6</td>
      <td>14</td>
      <td>588282</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2012-05-07 15:00:00</th>
      <td>31.060</td>
      <td>76</td>
      <td>262</td>
      <td>train</td>
      <td>2012-05-07 15:00:00</td>
      <td>0</td>
      <td>53</td>
      <td>186</td>
      <td>2</td>
      <td>25.42</td>
      <td>2</td>
      <td>19.9995</td>
      <td>1</td>
      <td>4.343805</td>
      <td>5.231109</td>
      <td>5.572154</td>
      <td>2012-05-07</td>
      <td>7</td>
      <td>5</td>
      <td>2012</td>
      <td>15</td>
      <td>0</td>
      <td>19</td>
      <td>588282</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2011-06-23 09:00:00</th>
      <td>34.850</td>
      <td>0</td>
      <td>0</td>
      <td>test</td>
      <td>2011-06-23 09:00:00</td>
      <td>0</td>
      <td>70</td>
      <td>0</td>
      <td>3</td>
      <td>30.34</td>
      <td>2</td>
      <td>8.9981</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2011-06-23</td>
      <td>23</td>
      <td>6</td>
      <td>2011</td>
      <td>9</td>
      <td>3</td>
      <td>25</td>
      <td>640662</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### 6. Training Models

We create a machine learning model to predict for casual and registered users separately and then combine them to generate the overall prediction for the counts.


```python
# instead of randomly splitting our training data 
# for cross validation, let's construct a framework that's more
# in line with how the data is divvied up for this competition
# (given first 19 days of each month, what is demand for remaining days)
# so, let's split our training data into 2 time contiguous datasets
# for fitting and validating our model (days 1-14 vs. days 15-19).

# also, since submissions are evaluated based on the
# root mean squared logarithmic error (RMSLE), let's replicate
# that computation as we test and tune our model.

train_df = all_df[all_df['data_set'] == 'train']
test_df = all_df[all_df['data_set'] == 'test']

def get_rmsle(y_pred, y_actual):
    diff = np.log(y_pred + 1) - np.log(y_actual + 1)
    mean_error = np.square(diff).mean()
    return np.sqrt(mean_error)

def custom_train_valid_split(data, cutoff_day=15):
    train = data[data['day'] <= cutoff_day]
    valid = data[data['day'] > cutoff_day]

    return train, valid

def prep_train_data(data, input_cols):
    X = data[input_cols].values
    y_r = data['registered_log'].values
    y_c = data['casual_log'].values

    return X, y_r, y_c

# predict on validation set & transform output back from log scale
def predict_on_validation_set(model, input_cols):
    
    train, valid = custom_train_valid_split(train_df)

    # prepare training & validation set
    X_train, y_train_r, y_train_c = prep_train_data(train, input_cols)
    X_valid, y_valid_r, y_valid_c = prep_train_data(valid, input_cols)

    model_r = model.fit(X_train, y_train_r)
    y_pred_r = np.exp(model_r.predict(X_valid)) - 1

    model_c = model.fit(X_train, y_train_c)
    y_pred_c = np.exp(model_c.predict(X_valid)) - 1

    y_pred_comb = np.round(y_pred_r + y_pred_c)
    y_pred_comb[y_pred_comb < 0] = 0

    y_actual_comb = np.exp(y_valid_r) + np.exp(y_valid_c) - 2

    rmsle = get_rmsle(y_pred_comb, y_actual_comb)
    return (y_pred_comb, y_actual_comb, rmsle)


# predict on test set & transform output back from log scale
def predict_on_test_set(model, input_cols):
    
    # prepare training set
    X_train, y_train_r, y_train_c = prep_train_data(train_df, input_cols)

    # prepare testing set
    X_test = test_df[input_cols].values
    
    model_c = model.fit(X_train, y_train_c)
    y_pred_c = np.exp(model_c.predict(X_test)) - 1

    model_r = model.fit(X_train, y_train_r)
    y_pred_r = np.exp(model_r.predict(X_test)) - 1
    
    # add casual & registered predictions together
    y_pred_comb = np.round(y_pred_r + y_pred_c)
    y_pred_comb[y_pred_comb < 0] = 0
    
    return y_pred_comb
```


```python
params = {'n_estimators': 1000, 'max_depth': 15, 'random_state': 0, 'min_samples_split' : 5, 'n_jobs': -1}
rf_model = RandomForestRegressor(**params)
rf_cols = [
    'weather', 'temp', 'atemp', 'windspeed',
    'workingday', 'season', 'holiday', 'sticky',
    'hour', 'dow', 'woy', 'peak'
    ]

(rf_pred, rf_actual, rf_rmsle) = predict_on_validation_set(rf_model, rf_cols)
```


```python
print(rf_rmsle)
```

    0.434625932998



```python
all_df[rf_cols].corr()
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
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>windspeed</th>
      <th>workingday</th>
      <th>season</th>
      <th>holiday</th>
      <th>sticky</th>
      <th>hour</th>
      <th>dow</th>
      <th>woy</th>
      <th>peak</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>weather</th>
      <td>1.000000</td>
      <td>-0.102640</td>
      <td>-0.105563</td>
      <td>0.026226</td>
      <td>0.042061</td>
      <td>-0.014524</td>
      <td>0.038472</td>
      <td>0.243523</td>
      <td>-0.020203</td>
      <td>-0.046424</td>
      <td>0.009692</td>
      <td>0.013506</td>
    </tr>
    <tr>
      <th>temp</th>
      <td>-0.102640</td>
      <td>1.000000</td>
      <td>0.987672</td>
      <td>-0.023125</td>
      <td>0.069153</td>
      <td>0.312025</td>
      <td>-0.101406</td>
      <td>-0.007074</td>
      <td>0.137603</td>
      <td>-0.036220</td>
      <td>0.198641</td>
      <td>0.044723</td>
    </tr>
    <tr>
      <th>atemp</th>
      <td>-0.105563</td>
      <td>0.987672</td>
      <td>1.000000</td>
      <td>-0.062336</td>
      <td>0.067594</td>
      <td>0.319380</td>
      <td>-0.101800</td>
      <td>0.004717</td>
      <td>0.133750</td>
      <td>-0.038918</td>
      <td>0.205561</td>
      <td>0.042167</td>
    </tr>
    <tr>
      <th>windspeed</th>
      <td>0.026226</td>
      <td>-0.023125</td>
      <td>-0.062336</td>
      <td>1.000000</td>
      <td>-0.001937</td>
      <td>-0.149773</td>
      <td>0.008593</td>
      <td>-0.187671</td>
      <td>0.137252</td>
      <td>0.003274</td>
      <td>-0.131613</td>
      <td>0.054581</td>
    </tr>
    <tr>
      <th>workingday</th>
      <td>0.042061</td>
      <td>0.069153</td>
      <td>0.067594</td>
      <td>-0.001937</td>
      <td>1.000000</td>
      <td>0.010879</td>
      <td>-0.091171</td>
      <td>0.536900</td>
      <td>0.002185</td>
      <td>-0.698028</td>
      <td>-0.025700</td>
      <td>0.207653</td>
    </tr>
    <tr>
      <th>season</th>
      <td>-0.014524</td>
      <td>0.312025</td>
      <td>0.319380</td>
      <td>-0.149773</td>
      <td>0.010879</td>
      <td>1.000000</td>
      <td>-0.109490</td>
      <td>0.095556</td>
      <td>-0.006117</td>
      <td>-0.007448</td>
      <td>0.814302</td>
      <td>-0.001289</td>
    </tr>
    <tr>
      <th>holiday</th>
      <td>0.038472</td>
      <td>-0.101406</td>
      <td>-0.101800</td>
      <td>0.008593</td>
      <td>-0.091171</td>
      <td>-0.109490</td>
      <td>1.000000</td>
      <td>-0.029345</td>
      <td>0.007158</td>
      <td>-0.049770</td>
      <td>0.104596</td>
      <td>-0.004713</td>
    </tr>
    <tr>
      <th>sticky</th>
      <td>0.243523</td>
      <td>-0.007074</td>
      <td>0.004717</td>
      <td>-0.187671</td>
      <td>0.536900</td>
      <td>0.095556</td>
      <td>-0.029345</td>
      <td>1.000000</td>
      <td>-0.186289</td>
      <td>-0.399949</td>
      <td>0.096672</td>
      <td>0.050274</td>
    </tr>
    <tr>
      <th>hour</th>
      <td>-0.020203</td>
      <td>0.137603</td>
      <td>0.133750</td>
      <td>0.137252</td>
      <td>0.002185</td>
      <td>-0.006117</td>
      <td>0.007158</td>
      <td>-0.186289</td>
      <td>1.000000</td>
      <td>-0.002893</td>
      <td>-0.005437</td>
      <td>0.124008</td>
    </tr>
    <tr>
      <th>dow</th>
      <td>-0.046424</td>
      <td>-0.036220</td>
      <td>-0.038918</td>
      <td>0.003274</td>
      <td>-0.698028</td>
      <td>-0.007448</td>
      <td>-0.049770</td>
      <td>-0.399949</td>
      <td>-0.002893</td>
      <td>1.000000</td>
      <td>0.009368</td>
      <td>-0.148325</td>
    </tr>
    <tr>
      <th>woy</th>
      <td>0.009692</td>
      <td>0.198641</td>
      <td>0.205561</td>
      <td>-0.131613</td>
      <td>-0.025700</td>
      <td>0.814302</td>
      <td>0.104596</td>
      <td>0.096672</td>
      <td>-0.005437</td>
      <td>0.009368</td>
      <td>1.000000</td>
      <td>-0.007311</td>
    </tr>
    <tr>
      <th>peak</th>
      <td>0.013506</td>
      <td>0.044723</td>
      <td>0.042167</td>
      <td>0.054581</td>
      <td>0.207653</td>
      <td>-0.001289</td>
      <td>-0.004713</td>
      <td>0.050274</td>
      <td>0.124008</td>
      <td>-0.148325</td>
      <td>-0.007311</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
params = {'n_estimators': 150, 'max_depth': 5, 'random_state': 0, 'min_samples_leaf' : 10, 'learning_rate': 0.1, 'subsample': 0.7, 'loss': 'ls'}
gbm_model = GradientBoostingRegressor(**params)
gbm_cols = [
    'weather', 'temp', 'atemp', 'humidity', 'windspeed',
    'holiday', 'workingday', 'season',
    'hour', 'dow', 'year', 'ideal', 'count_season',
]

(gbm_pred, gbm_actual, gbm_rmsle) = predict_on_validation_set(gbm_model, gbm_cols)
```


```python
print(gbm_rmsle)
```

    0.313091436534



```python
all_df[gbm_cols].corr()
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
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>season</th>
      <th>hour</th>
      <th>dow</th>
      <th>year</th>
      <th>ideal</th>
      <th>count_season</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>weather</th>
      <td>1.000000</td>
      <td>-0.102640</td>
      <td>-0.105563</td>
      <td>0.418130</td>
      <td>0.026226</td>
      <td>0.038472</td>
      <td>0.042061</td>
      <td>-0.014524</td>
      <td>-0.020203</td>
      <td>-0.046424</td>
      <td>-0.019157</td>
      <td>-0.145407</td>
      <td>-0.051863</td>
    </tr>
    <tr>
      <th>temp</th>
      <td>-0.102640</td>
      <td>1.000000</td>
      <td>0.987672</td>
      <td>-0.069881</td>
      <td>-0.023125</td>
      <td>-0.101406</td>
      <td>0.069153</td>
      <td>0.312025</td>
      <td>0.137603</td>
      <td>-0.036220</td>
      <td>0.040913</td>
      <td>0.727266</td>
      <td>0.705172</td>
    </tr>
    <tr>
      <th>atemp</th>
      <td>-0.105563</td>
      <td>0.987672</td>
      <td>1.000000</td>
      <td>-0.051918</td>
      <td>-0.062336</td>
      <td>-0.101800</td>
      <td>0.067594</td>
      <td>0.319380</td>
      <td>0.133750</td>
      <td>-0.038918</td>
      <td>0.039222</td>
      <td>0.701874</td>
      <td>0.701434</td>
    </tr>
    <tr>
      <th>humidity</th>
      <td>0.418130</td>
      <td>-0.069881</td>
      <td>-0.051918</td>
      <td>1.000000</td>
      <td>-0.290105</td>
      <td>0.014029</td>
      <td>0.014316</td>
      <td>0.150625</td>
      <td>-0.276498</td>
      <td>-0.035233</td>
      <td>-0.083546</td>
      <td>-0.141678</td>
      <td>0.113724</td>
    </tr>
    <tr>
      <th>windspeed</th>
      <td>0.026226</td>
      <td>-0.023125</td>
      <td>-0.062336</td>
      <td>-0.290105</td>
      <td>1.000000</td>
      <td>0.008593</td>
      <td>-0.001937</td>
      <td>-0.149773</td>
      <td>0.137252</td>
      <td>0.003274</td>
      <td>-0.008740</td>
      <td>-0.051489</td>
      <td>-0.113048</td>
    </tr>
    <tr>
      <th>holiday</th>
      <td>0.038472</td>
      <td>-0.101406</td>
      <td>-0.101800</td>
      <td>0.014029</td>
      <td>0.008593</td>
      <td>1.000000</td>
      <td>-0.091171</td>
      <td>-0.109490</td>
      <td>0.007158</td>
      <td>-0.049770</td>
      <td>0.006293</td>
      <td>-0.054138</td>
      <td>-0.146902</td>
    </tr>
    <tr>
      <th>workingday</th>
      <td>0.042061</td>
      <td>0.069153</td>
      <td>0.067594</td>
      <td>0.014316</td>
      <td>-0.001937</td>
      <td>-0.091171</td>
      <td>1.000000</td>
      <td>0.010879</td>
      <td>0.002185</td>
      <td>-0.698028</td>
      <td>-0.007959</td>
      <td>0.023068</td>
      <td>0.044535</td>
    </tr>
    <tr>
      <th>season</th>
      <td>-0.014524</td>
      <td>0.312025</td>
      <td>0.319380</td>
      <td>0.150625</td>
      <td>-0.149773</td>
      <td>-0.109490</td>
      <td>0.010879</td>
      <td>1.000000</td>
      <td>-0.006117</td>
      <td>-0.007448</td>
      <td>-0.010742</td>
      <td>0.156455</td>
      <td>0.663537</td>
    </tr>
    <tr>
      <th>hour</th>
      <td>-0.020203</td>
      <td>0.137603</td>
      <td>0.133750</td>
      <td>-0.276498</td>
      <td>0.137252</td>
      <td>0.007158</td>
      <td>0.002185</td>
      <td>-0.006117</td>
      <td>1.000000</td>
      <td>-0.002893</td>
      <td>-0.003867</td>
      <td>0.113745</td>
      <td>-0.008248</td>
    </tr>
    <tr>
      <th>dow</th>
      <td>-0.046424</td>
      <td>-0.036220</td>
      <td>-0.038918</td>
      <td>-0.035233</td>
      <td>0.003274</td>
      <td>-0.049770</td>
      <td>-0.698028</td>
      <td>-0.007448</td>
      <td>-0.002893</td>
      <td>1.000000</td>
      <td>0.000977</td>
      <td>-0.009208</td>
      <td>-0.014472</td>
    </tr>
    <tr>
      <th>year</th>
      <td>-0.019157</td>
      <td>0.040913</td>
      <td>0.039222</td>
      <td>-0.083546</td>
      <td>-0.008740</td>
      <td>0.006293</td>
      <td>-0.007959</td>
      <td>-0.010742</td>
      <td>-0.003867</td>
      <td>0.000977</td>
      <td>1.000000</td>
      <td>0.000788</td>
      <td>-0.009706</td>
    </tr>
    <tr>
      <th>ideal</th>
      <td>-0.145407</td>
      <td>0.727266</td>
      <td>0.701874</td>
      <td>-0.141678</td>
      <td>-0.051489</td>
      <td>-0.054138</td>
      <td>0.023068</td>
      <td>0.156455</td>
      <td>0.113745</td>
      <td>-0.009208</td>
      <td>0.000788</td>
      <td>1.000000</td>
      <td>0.462633</td>
    </tr>
    <tr>
      <th>count_season</th>
      <td>-0.051863</td>
      <td>0.705172</td>
      <td>0.701434</td>
      <td>0.113724</td>
      <td>-0.113048</td>
      <td>-0.146902</td>
      <td>0.044535</td>
      <td>0.663537</td>
      <td>-0.008248</td>
      <td>-0.014472</td>
      <td>-0.009706</td>
      <td>0.462633</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



### 7. Stacking

We can combine the predictions of two or more models to create a meta prediction.

It's much like cross validation. Take 5-fold stacking as an example. First we split the training data into 5 folds. Next we will do 5 iterations. In each iteration, train every base model on 4 folds and predict on the hold-out fold. **We have to keep the predictions on the testing data as well**. This way, in each iteration every base model will make predictions on 1 fold of the training data and all of the testing data. After 5 iterations we will obtain a matrix of shape `#(samples in training data) X #(base models)`. This matrix is then fed to the stacker (it’s just another model) in the second level. After the stacker is fitted, use the predictions on testing data by base models (**each base model is predicts on the test data, since there are 5 base models we will get 5 predictions on thesame test data, therefore we have to take an average to obtain a matrix of the same shape**) as the input for the stacker and obtain our final predictions.

![stacking](stacking.jpg)

#### 7.1 Manual Stacking
We can stack the predictions of two or more different models in a pre-defined weighted mechanism to get the final prediction. Instead of relying on the prediction of a single model we average out the predictions from two or more models.


```python
# the blend gives a better score on the leaderboard, even though it does not on the validation set
y_pred = np.round(.2*rf_pred + .8*gbm_pred)
print(get_rmsle(y_pred, rf_actual))
```

    0.316017221761



```python
rf_pred = predict_on_test_set(rf_model, rf_cols)
gbm_pred = predict_on_test_set(gbm_model, gbm_cols)

y_pred = np.round(.2*rf_pred + .8*gbm_pred)
```


```python
# output predictions for submission
submit_manual_blend_df = test_df[['datetime', 'count']].copy()
submit_manual_blend_df['count'] = y_pred
```


```python
submit_manual_blend_df.head(5)
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
      <th>datetime</th>
      <th>count</th>
    </tr>
    <tr>
      <th>datetime</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2011-01-20 00:00:00</th>
      <td>2011-01-20 00:00:00</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>2011-01-20 01:00:00</th>
      <td>2011-01-20 01:00:00</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>2011-01-20 02:00:00</th>
      <td>2011-01-20 02:00:00</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2011-01-20 03:00:00</th>
      <td>2011-01-20 03:00:00</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2011-01-20 04:00:00</th>
      <td>2011-01-20 04:00:00</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
submit_manual_blend_df.to_csv('output/submit_manual_blend.csv', index=False)
```

#### 7.1 Stacking  with Linear Regression

In the previous section we manually assigned weights to the two classifiers and blended the predictions. Here we will train the same two classifiers with different input cols on the data and then combine the predictions of the two classifiers using a LinearRegressor. The LinearRegressor acts as a meta-classifier which learnes what should be the weighted combination of the two level 0 classifiers.


```python
# Level 0 RandomForestRegressor
rf_params = {'n_estimators': 1000, 'max_depth': 15, 'random_state': 0, 'min_samples_split' : 5, 'n_jobs': -1}
rf_model = RandomForestRegressor(**rf_params)
rf_cols = [
    'weather', 'temp', 'atemp', 'windspeed',
    'workingday', 'season', 'holiday', 'sticky',
    'hour', 'dow', 'woy', 'peak'
    ]
```


```python
# Level 0 GradientBoostingRegressor
gbm_params = {'n_estimators': 150, 'max_depth': 5, 'random_state': 0, 'min_samples_leaf' : 10, 'learning_rate': 0.1, 'subsample': 0.7, 'loss': 'ls'}
gbm_model = GradientBoostingRegressor(**gbm_params)
gbm_cols = [
    'weather', 'temp', 'atemp', 'humidity', 'windspeed',
    'holiday', 'workingday', 'season',
    'hour', 'dow', 'year', 'ideal', 'count_season',
]
```


```python
clf_input_cols = [rf_cols, gbm_cols]
clfs = [rf_model, gbm_model]
```


```python
# Create train and test sets for blending and Pre-allocate the data
blend_train = np.zeros((train_df.shape[0], len(clfs)))
blend_test = np.zeros((test_df.shape[0], len(clfs)))
```


```python
# For each classifier, we train the classifier with its corresponding input_cols 
# and record the predictions on the train and the test set
for clf_index, (input_cols, clf) in enumerate(zip(clf_input_cols, clfs)):
    
    # prepare training and validation set
    X_train, y_train_r, y_train_c = prep_train_data(train_df, input_cols)
    
    # prepare testing set
    X_test = test_df[input_cols].values
    
    model_r = clf.fit(X_train, y_train_r)
    y_pred_train_r = np.exp(model_r.predict(X_train)) - 1
    y_pred_test_r = np.exp(model_r.predict(X_test)) - 1

    model_c = clf.fit(X_train, y_train_c)
    y_pred_train_c = np.exp(model_c.predict(X_train)) - 1
    y_pred_test_c = np.exp(model_c.predict(X_test)) - 1

    y_pred_train_comb = np.round(y_pred_train_r + y_pred_train_c)
    y_pred_train_comb[y_pred_train_comb < 0] = 0
    
    y_pred_test_comb = np.round(y_pred_test_r + y_pred_test_c)
    y_pred_test_comb[y_pred_test_comb < 0] = 0
    
    blend_train[:, clf_index] = y_pred_train_comb
    blend_test[:, clf_index] = y_pred_test_comb
```


```python
# Level 1 Belending Classifier using LinearRegression
from sklearn.linear_model import LinearRegression
bclf = LinearRegression(fit_intercept=False)
bclf.fit(blend_train, train_df['count'])
```




    LinearRegression(copy_X=True, fit_intercept=False, n_jobs=1, normalize=False)




```python
# What is the weighted combination of the base classifiers?
bclf.coef_
```




    array([ 0.31355465,  0.73048534])



> We observe that the meta-learner LinearRegression has assigned weights 0.3 to the RandomForestRegressor and 0.7 to the GradientBoostingRegressor similar to the ones we did manually in the previous step.


```python
# Stacked and Blending predictions
y_pred_blend = np.round(bclf.predict(blend_test))
```


```python
# R^2 score
bclf.score(blend_train, train_df['count'])
```




    0.96562432974162338




```python
# output predictions for submission
submit_stack_blend_df = test_df[['datetime', 'count']].copy()
submit_stack_blend_df['count'] = y_pred_blend
```


```python
submit_stack_blend_df.head(5)
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
      <th>datetime</th>
      <th>count</th>
    </tr>
    <tr>
      <th>datetime</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2011-01-20 00:00:00</th>
      <td>2011-01-20 00:00:00</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>2011-01-20 01:00:00</th>
      <td>2011-01-20 01:00:00</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>2011-01-20 02:00:00</th>
      <td>2011-01-20 02:00:00</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2011-01-20 03:00:00</th>
      <td>2011-01-20 03:00:00</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2011-01-20 04:00:00</th>
      <td>2011-01-20 04:00:00</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
submit_stack_blend_df.to_csv('output/submit_stack_blend.csv', index=False)
```

**References:**

Blogs without which this notebook would not have been possible

+ [1] [https://www.analyticsvidhya.com/blog/2015/06/solution-kaggle-competition-bike-sharing-demand/](https://www.analyticsvidhya.com/blog/2015/06/solution-kaggle-competition-bike-sharing-demand/)
+ [2][https://www.kaggle.com/benhamner/bike-rentals-by-time-and-temperature](https://www.kaggle.com/benhamner/bike-rentals-by-time-and-temperature)
+ [3][https://github.com/logicalguess/kaggle-bike-sharing-demand](https://github.com/logicalguess/kaggle-bike-sharing-demand)
+ [4][http://dataviztalk.blogspot.com/2016/02/how-to-add-jitter-to-plot-using-pythons.html](http://dataviztalk.blogspot.com/2016/02/how-to-add-jitter-to-plot-using-pythons.html)
+ [5][https://www.kaggle.com/anuragreddy333/data-vizualization/code](https://www.kaggle.com/anuragreddy333/data-vizualization/code)
+ [6][https://dnc1994.com/2016/05/rank-10-percent-in-first-kaggle-competition-en/]


```python

```

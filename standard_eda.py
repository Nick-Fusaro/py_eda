#TODO: Finish converting this into PySpark
#TODO: Add docstrings
#TODO: Look up requirements to run this in both databricks and in test.
#TODO: Think harder about whether this should work this way (hell of a lot of .collect() in there)

import glob as glob
import os
from collections import Counter

import pyspark.sql.functions as F
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext


"""
TEST REQUIREMENTS START:
"""
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext

conf = SparkConf().setAppName('appName').setMaster('local')
sc = SparkContext(conf=conf)
sc.setLogLevel("WARN")
sql = SQLContext(sc)
df = sql.read.csv("sample_data.csv", inferSchema = True, 
                  header=True, nullValue="NULL")
"""
TEST REQUIREMENTS END
"""


def create_meta_data(df):
    """Returns a spark DataFrame with datatype columns.
    
    Arguments:
        df {DataFrame} 
    """
    return sc.parallelize(df.dtypes).toDF(["name", "dtype"])

def select_dtypes(df, include=None, exclude = None):
    """Selects columns from a DataFrame by type, 
    including, excluding, or both. Eligible types are
    "string", "int", "float", "double", "datetime", 
    "struct", "array", and probably more.

    TODO: This should probably throw if you put in a datatype that's not valid.
    
    Arguments:
        df {DataFrame} 
    
    Keyword Arguments:
        include {[List]} -- List of datatypes to include. (default: {['None']})
        exclude {[List]} -- List of datatypes to exclude. (default: {['None']})
    """
    dtypes = create_meta_data(df)
    exclude_str = str(exclude).replace('[','(').replace(']',')')
    include_str = str(include).replace('[','(').replace(']',')')
    if include and exclude:
        condition_inc = "dtype in " + include_str
        condition_exc =  "dtype not in " + exclude_str
        return dtypes.filter(condition_inc).filter(condition_exc).rdd.keys().collect()
    elif include:
        condition_inc = "dtype in " + include_str
        return dtypes.filter(condition_inc).rdd.keys().collect()
    elif exclude:
        condition_exc =  "dtype not in " + exclude_str
        return dtypes.filter(condition_exc).rdd.keys().collect()

def agg_all_columns(df, aggfunc):
    """Apply an aggregate function that works on DataFrame.Column
    objects to all columns in a dataframe.
    
    Arguments:
        df {DataFrame} -- Dataframe to aggregate.
        aggfunc {Function} -- Function to be applied.
    """
    #df is a dataframe and agg has to be a pyspark sql function (F.min)
    exprs = [aggfunc(x) for x in df.columns]
    return df.agg(*exprs).toDF(*df.columns) #return original column names

def filter_aggregates(df, aggfunc, low_bound = None, up_bound = None):
    """Aggregate all columns in a dataframe, and then return columns
    which violate the lower or upper bounds.
    
    Arguments:
        df {DataFrame} -- Dataframe to filter.
        aggfunc {function} -- Aggregate to filter on.
    
    Keyword Arguments:
        low_bound {Numeric} -- Lower bound of filter (default: {None})
        up_bound {Numeric} -- Upper bound of filter (default: {None})

    Returns:
        col_list {List} -- A list of columns for which the criterion is 
        violated.
    """
    #Return a list of columns where the result of aggfunc IS NOT between low_bound and up_bound 
    aggs = (agg_all_columns(df, aggfunc).rdd.collect()[0].asDict())
    if low_bound is not None and up_bound is not None :
        return [k for (k, v) in aggs.items() if  low_bound >= v or  v >= up_bound]
    elif low_bound is not None:
        return [k for (k, v) in aggs.items() if  low_bound >= v]
    elif up_bound is not None:
        return [k for (k, v) in aggs.items() if  v >= up_bound]
    else:
        raise ValueError("Remember to specify at least one of (low_bound, up_bound)")

def get_dim(df):
    """Returns dimensions of spark DataFrame
    
    Arguments:
        df {DataFrame} 
    """
    return (df.count(), len(df.columns))

def find_non_cata_numeric(df,i):
    """Finds numeric columns that might be better as categorical objects 
    (i.e, have less than i distinct values)
    
    Arguments:
        df {DataFrame} -- 
        i {int} -- Threshold for converting a column to a category.
    """
    non_cat_list = select_dtypes(df, exclude=['string'])
    less_i_list = filter_aggregates(df[non_cat_list], F.countDistinct, low_bound=i)
    print("Warning - the following numerical columns might be better suited as categorical objects")
    print(less_i_list)
    return less_i_list

def pctMissing(df):
    """Returns the percentage of values in each column
    which are missing (null).
    
    Arguments:
        df {DataFrame}
    """
    #TODO: Drop or bin column if missing above threshold
    N_rows, _ = get_dim(df)
    get_perc_na = lambda x: 100 * F.sum(F.isnull(x).cast(DoubleType()))/N_rows
    return agg_all_columns(df, get_perc_na)
    
meta_data = create_meta_data(df)





original_dim = get_dim(df)

print("There are " + str(original_dim[1]) + " columns" +
      " and "  + str(original_dim[0])+ " rows in the original data")

#Create a list of categorical variables

cat_list= select_dtypes(df, include=['string'])

# #Delete any categorical vars from the list you
# #know you arent interested in
# del cat_list[('variable')]

#Create a list of float columns
float_list= select_dtypes(df, include=['float'])
# #Create a list of int columns
int_list= select_dtypes(df, include=['int'])

num_list = float_list+int_list
# #Checks wheher any float columns have a max of 0 and are useless


max_zero_list = filter_aggregates(df.select(*num_list), F.max, low_bound = 0)
print("Warning - the following numeric columns have a maximum of 0:")
print(max_zero_list)

df = df.drop(*max_zero_list)
float_list = [x for x in num_list if x not in max_zero_list]
new_dim = get_dim(df)
print( "There were "  + str(original_dim[1]) + " columns and now are " + str(new_dim[1]) + " columns")



find_non_cata_numeric(df, 3)
from pyspark.sql.types import DoubleType


#TODO: Sell Erik on XGBoost as our default method because then we don't have to
# deal with nulls. Plus it does the binning automatically and better than we can.


# NOTE: Dropped the standard scaler function -- MLLib is okay to do that with.


# TODO: Decide if this should be in at all, it's dumb hard to get it into spark.

# #Function to do frequency analysis
# def freq_iter(df, cols):
#     dflist = []
#     for i in cols:
#         series = df[i]
#         #Count the discrete values
#         type_count = [[k,v] for k,v in Counter(series).items()]
#         # Find the count of all discrete elements 
#         j=pd.DataFrame.from_records(type_count, 
#                                     index=range(len(type_count)), 
#                                     columns=['value','frequency'])
#         j['percent'] = j.frequency/j.frequency.sum() 
#         j['cumulative_frequency'] = j.frequency.cumsum()
#         j['cumulative_percent'] = j.cumulative_frequency/j.frequency.sum()
#         j.columns = [series.name + '_' + i for i in j.columns]
#         dflist.append(j)
#     return dflist

# #Create a list with the frequency analysis function and displays the contents 
# freq_iter_list = freq_iter(df, cat_list)
# for p in freq_iter_list: 
#      display (p.iloc[:,[0,1,2]])



## AAH THIS IS HARD TOO EVEN IF IT IS USEFUL
# TODO: Add this function anyway
# def add_quantiles(data, columns, suffix, quantiles=4, labels=None):
#         """ For each column name in columns, create a new categorical column with
#             the same name as colum, with the suffix specified added, that
#             specifies the quantile of the row in the original column using
#             pandas.qcut().

#             Input
#             _____
#             data:
#                 pandas dataframe
#             columns:
#                 list of column names in `data` for which this function is to create
#                 quantiles
#             suffix:
#                 string suffix for new column names ({`suffix`}_{collumn_name})
#             labels:
#                 list of labels for each quantile (should have length equal to `quantiles`)

#             Output
#             ______
#             pandas dataframe containing original columns, plus new columns with quantile
#             information for specified columns.
#         """
#         d = data.copy()
#         quantile_labels = labels or [
#             '{:.0f}%'.format(i*100/quantiles) for i in range(1, quantiles+1)
#         ]
#         for column in columns:
#             d[colname(column, suffix)] = pd.qcut(d[column], quantiles, labels=quantile_labels)
#         return d

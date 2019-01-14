#data preprocessing

#loading libraries

import numpy as np
import pandas as pd
import matplotlib as mplt
from matplotlib import pyplot

df = pd.read_csv("co2_emission.csv")
other_path = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/auto.csv"
df1 = pd.read_csv(other_path, header=None)

df.replace("?", np.nan, inplace = True)
#view the dataset
print(df.head())
print(df.tail())
#adding column names to the dataset
df.columns=['symboling','normalized-losses','make','fuel-type','aspiration','num-of-doors','body-style','drive-wheels',
                            'engine-location','wheel-base','length','width','height','curb-weight','engine-type','num-of-cylinders','engine-size',
                            'fuel-system','bore','stroke','compression-ratio','horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price'];


#descriptive statistics
print(df.describe())
#only shows count, unique, freq and top on all variable
print(df.describe(include="all"))

#fining missing data
missing_data = df.isnull()

#finding missing columns
for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")

#converting a datatype of a column
mean_normalized_losses = df["normalized-losses"].astype(float).mean()
print(mean_normalized_losses)
df["normalized-losses"].replace(np.nan, mean_normalized_losses,inplace=True)
print(df["normalized-losses"])

#fixing missing values with mean value
mean_bores = df["bore"].astype(float).mean()
print(mean_bores)
df["bore"].replace(np.nan, mean_bores,inplace=True)
print(df["bore"])

mean_stroke = df["stroke"].astype(float).mean()
print(mean_stroke)
df["stroke"].replace(np.nan, mean_stroke,inplace=True)
print(df["stroke"])

mean_horsepower = df["horsepower"].astype(float).mean()
print(mean_bores)
df["horsepower"].replace(np.nan, mean_horsepower,inplace=True)
print(df["horsepower"])

mean_peak_rpm = df["peak-rpm"].astype(float).mean()
print(mean_peak_rpm)
df["peak-rpm"].replace(np.nan, mean_peak_rpm,inplace=True)
print(df["peak-rpm"])

mean_price = df["price"].astype(float).mean()
print(mean_price)
df["price"].replace(np.nan, mean_price,inplace=True)
print(df["price"])

#resetting index after dropping na
df.reset_index(drop=True, inplace=True)

#to find the most frequenct works in string or object  rtpe, use value_counts().idmax()
frequent_occured = df['num-of-doors'].value_counts().idxmax()
df["num-of-doors"].replace(np.nan, frequent_occured,inplace=True)
print(df["num-of-doors"])

#Converting to required data type
df[["bore", "stroke","price","peak-rpm"]] = df[["bore", "stroke","price","peak-rpm"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")

print(df.dtypes)

#Standardisation of data
df["highway-mpg"] = 235/df["highway-mpg"]
df.rename(columns={"highway-mpg":"highway-L/100km"},inplace=True)
print(df.columns)

#Normalisation
# #Simple feature Scaling
df["length"]= df["length"]/df["length"].max()
print(df["length"])

#Min- Max method
df["width"]= (df["width"]-df["width"].min())/(df["width"].min()-df["width"].max())
print(df["width"])

#z test method
df["height"]= (df["height"]-df["height"].mean())/(df["height"].std())
print(df["height"])

#binning the column and plotting a bar plot
binwidth = 4
df[["horsepower"]] = df[["horsepower"]].astype("int")
bins_for_horsepower = np.linspace(min(df["horsepower"]),max(df["horsepower"]),binwidth)
bin_values = ['low','medium','high']
df["binned_horsepower"]= pd.cut(df["horsepower"],bins_for_horsepower, labels=bin_values,include_lowest=True)
print(df[['horsepower','binned_horsepower']].head(20))

pyplot.bar(bin_values,df["binned_horsepower"].value_counts())

mplt.pyplot.xlabel("horsepower")
mplt.pyplot.ylabel("count")
mplt.pyplot.title("binned_horsepower")
pyplot.show()

#Creating dummies for a coulumns
dummy_variables = pd.get_dummies(df['fuel-type'])
print(dummy_variables.head())

dummy_variables.rename(columns={'fuel_type-diesel':'diesel','fuel-type-gas':'gas'},inplace=True)

# merge data frame "df" and "dummy_variable_1"
df = pd.concat([df, dummy_variables], axis=1)
print(df.head())




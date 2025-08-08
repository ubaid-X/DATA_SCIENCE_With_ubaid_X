import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2

# Load the Titanic dataset
titanic = sns.load_dataset('titanic')

# Handling Missing Values
# Impute missing values in 'age' with the mean
imputer = SimpleImputer(strategy='mean')
titanic['age'] = imputer.fit_transform(titanic[['age']])
# Assume 'deck' has too many missing values and drop it
titanic.drop(columns=['deck'], inplace=True)

# Outlier Detection and Removal
# Detect and remove outliers in 'fare' based on the Interquartile Range (IQR)
Q1 = titanic['fare'].quantile(0.25)
Q3 = titanic['fare'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
titanic = titanic[(titanic['fare'] >= lower_bound) & (titanic['fare'] <= upper_bound)]

# Normalization
# Normalize 'fare' to have values between 0 and 1
scaler_min_max = MinMaxScaler()
titanic['fare_normalized'] = scaler_min_max.fit_transform(titanic[['fare']])

# Standardization
# Standardize 'age' to have a mean of 0 and a standard deviation of 1
scaler_std = StandardScaler()
titanic['age_standardized'] = scaler_std.fit_transform(titanic[['age']])

# Binning
# Transform 'age' into three discrete categories
titanic['age_binned'] = pd.cut(titanic['age'], bins=[0, 18, 60, 100], labels=["Child", "Adult", "Senior"])

# Feature Engineering
# Create a new feature 'family_size' from 'sibsp' and 'parch'
titanic['family_size'] = titanic['sibsp'] + titanic['parch'] + 1

# Feature Selection
# Select the top 3 features that have the highest correlation with 'survived'
X = titanic[['pclass', 'age', 'sibsp', 'parch', 'fare_normalized']]
y = titanic['survived']
selector = SelectKBest(score_func=chi2, k=3)
X_selected = selector.fit_transform(X, y)

# Encoding Categorical Variables
# Convert 'sex' into a numerical format using Label Encoding
label_encoder = LabelEncoder()
titanic['sex_encoded'] = label_encoder.fit_transform(titanic['sex'])

# Convert 'embarked' into binary columns using One-Hot Encoding
one_hot_encoder = OneHotEncoder()
encoded_embarked = one_hot_encoder.fit_transform(titanic[['embarked']]).toarray()
embarked_columns = one_hot_encoder.get_feature_names_out(['embarked'])
titanic = titanic.join(pd.DataFrame(encoded_embarked, columns=embarked_columns))

# Data Splitting
# Split the data into training and testing sets
X = titanic[['pclass', 'sex_encoded', 'age_standardized', 'sibsp', 'parch', 'fare_normalized', 'family_size']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Now, the dataset is ready for model training
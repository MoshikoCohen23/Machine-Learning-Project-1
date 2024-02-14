#import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sc
import matplotlib.pyplot as plt
from statsmodels.graphics.mosaicplot import mosaic

# dataset
df = pd.read_excel('CarInsurance_Xy_train.xlsx')
df = df.replace(-999, np.nan)
nulls = df.isnull().sum()

# variable probability
outcome_prob = df['OUTCOME'].value_counts() / df['OUTCOME'].shape[0]
age_prob = df['AGE'].value_counts() / df['AGE'].shape[0]
gender_prob = df['GENDER'].value_counts() / df['GENDER'].shape[0]
driving_exp_prob = df['DRIVING_EXPERIENCE'].value_counts() / df['DRIVING_EXPERIENCE'].shape[0]
education_prob = df['EDUCATION'].value_counts() / df['EDUCATION'].shape[0]
income_prob = df['INCOME'].value_counts() / df['INCOME'].shape[0]
credit_score_prob = df['CREDIT_SCORE'].value_counts() / df['CREDIT_SCORE'].shape[0]
vehicle_ownership_prob = df['VEHICLE_OWNERSHIP'].value_counts() / df['VEHICLE_OWNERSHIP'].shape[0]
vehicle_year_prob = df['VEHICLE_YEAR'].value_counts() / df['VEHICLE_YEAR'].shape[0]
married_prob = df['MARRIED'].value_counts() / df['MARRIED'].shape[0]
children_prob = df['CHILDREN'].value_counts() / df['CHILDREN'].shape[0]
postal_code_prob = df['POSTAL_CODE'].value_counts() / df['POSTAL_CODE'].shape[0]
annual_mileage_prob = df['ANNUAL_MILEAGE'].value_counts() / df['ANNUAL_MILEAGE'].shape[0]
vehicle_type_prob = df['VEHICLE_TYPE'].value_counts() / df['VEHICLE_TYPE'].shape[0]
speeding_violations_prob = df['SPEEDING_VIOLATIONS'].value_counts() / df['SPEEDING_VIOLATIONS'].shape[0]
past_accidents_prob = df['PAST_ACCIDENTS'].value_counts() / df['PAST_ACCIDENTS'].shape[0]

#graph variable
plot_annual =  sns.countplot(x='ANNUAL_MILEAGE', data=df)
plot_annual.set_title('Annual Milegae Frequency')
plt.xlabel('annual milegae')
plt.ylabel('frequency')
plt.show()

plt.title('Credit Score histogram')
sns.histplot(data=df, x='CREDIT_SCORE', kde=True)
plt.title("Credit Score Frequency")
plt.xlabel('credit score')
plt.ylabel('frequency')
plt.show()

sns.histplot(data=df['PAST_ACCIDENTS'], discrete=True)
plt.xlabel("past accidents")
plt.ylabel('frequency')
plt.title("Past Accidents Frequency")
plt.show()

sns.histplot(data=df['SPEEDING_VIOLATIONS'], discrete=True)
plt.xlabel("speeding violations")
plt.ylabel('frequency')
plt.title("Speeding Violations Frequency")
plt.show()

plot_outcome =  sns.countplot(x='OUTCOME', data=df, color='darkblue')
plot_outcome.set_title('Outcome Frequency')
plt.ylabel('frequency')
plt.xlabel("outcome")


#connection between variable
table_cor = df.drop('OUTCOME',1).corr()

table1 = df.groupby(['SPEEDING_VIOLATIONS','PAST_ACCIDENTS','DRIVING_EXPERIENCE'], as_index=False).size()
driving_exp_colors = { '0-9y': 'tab:red', '10-19y': 'tab:blue', '20-29y': 'tab:green', '30y+': 'tab:purple' }
plt.scatter(table1["SPEEDING_VIOLATIONS"], table1["PAST_ACCIDENTS"], s=table1["size"] * 10, c=table1['DRIVING_EXPERIENCE'].map(driving_exp_colors))
plt.title('The Connection Between Speeding Violations, Past Accidents by Driving Experience and Frequency')
plt.xlabel('Speeding Violations')
plt.ylabel('Past Accidents')
plt.legend()

sns.boxplot(x=df['EDUCATION'], y=df['CREDIT_SCORE'], palette="Set3")
plt.title('Credit Score by Education')
plt.xlabel('education')
plt.ylabel('credit score')

sns.boxplot(x=df['POSTAL_CODE'], y=df['CREDIT_SCORE'], palette="Set3")
plt.title('Credit Score by Postal Code')
plt.xlabel('postal code')
plt.ylabel('credit score')

sns.boxplot(x=df['INCOME'], y=df['CREDIT_SCORE'], palette="Set3")
plt.title('Credit Score by Income')
plt.xlabel('income')
plt.ylabel('credit score')

sns.boxplot(x=df['GENDER'], y=df['SPEEDING_VIOLATIONS'], palette="Set3")
plt.title('Speeding Violations by Gender')
plt.xlabel('gender')
plt.ylabel('speeding violations')

sns.boxplot(x=df['GENDER'], y=df['PAST_ACCIDENTS'], palette="Set3")
plt.title('Past Accidents by Gender')
plt.xlabel('gender')
plt.ylabel('past accidents')

sns.boxplot(x=df['CHILDREN'], y=df['ANNUAL_MILEAGE'], palette="Set3")
plt.xlabel('Children')
plt.ylabel('Annual mileage')
plt.title('Annual Mileage by Children')

sns.boxplot(x=df['MARRIED'], y=df['ANNUAL_MILEAGE'], palette="Set3")
plt.xlabel('Married')
plt.ylabel('Annual mileage')
plt.title('Annual Mileage by Married')

sns.boxplot(x=df['PAST_ACCIDENTS'], y=df['ANNUAL_MILEAGE'], palette="Set3")
plt.xlabel('past accidents')
plt.ylabel('annual mileage')
plt.title('Annual Mileage by Past Accidents')

sns.boxplot(x=df['VEHICLE_OWNERSHIP'], y=df['CREDIT_SCORE'], palette="Set3")
plt.title('Credit Score by Vehicle Ownership')
plt.xlabel('vehicle ownership')
plt.ylabel('credit score')

### data quality
sns.boxplot(y=df['SPEEDING_VIOLATIONS'], palette="Set3")
plt.xlabel('speeding violations')
plt.ylabel('count')
plt.title('Speeding Violations BoxPlot')

sns.boxplot(y=df['PAST_ACCIDENTS'], palette="Set3")
plt.xlabel('past accidents')
plt.ylabel('count')
plt.title('Past Accidents BoxPlot')

##### variable to outcome
## remove variable
sns.histplot(data=df, x='ANNUAL_MILEAGE',hue='OUTCOME', kde=True)
plt.xlabel('annual mileage')
plt.ylabel('frequency')
plt.title("The Connection between Annual Mileage to Insurance Claim Via PDF")

mosaic(data=df, index=['VEHICLE_TYPE','OUTCOME'],gap=0.02,title='The Connection Between Vehicle Type to Insurance Claim',
       labelizer=lambda k:{('sports car','0'):17,('sports car','1'):9,('sedan','0'):192,('sedan','1'):159}[k])
plt.show()

OUTCOME_1 = df[df['OUTCOME'] ==1].groupby('DRIVING_EXPERIENCE').size()
OUTCOME_0 = df[df['OUTCOME'] ==0].groupby('DRIVING_EXPERIENCE').size()
# Set position of bar on X axis
br1 = np.arange(len(OUTCOME_1))
br2 = [x + 0.25 for x in br1]
# Make the plot
plt.bar(br1, OUTCOME_1, color='lightblue', width=0.25,
        edgecolor='grey', label='OUTCOME_1')
plt.bar(br2, OUTCOME_0, color='lightgreen', width=0.25,
        edgecolor='grey', label='OUTCOME_0')
# Adding Xticks
plt.xlabel('Driving experience', fontweight='bold', fontsize=15)
plt.ylabel('Count', fontweight='bold', fontsize=15)
plt.xticks([r + 0.12 for r in range(len(OUTCOME_1))],
           ['0-9y', '10-19y', '20-29y', '30y+'])
plt.title('The Connection Between Driving Experience To Outcome')
plt.legend()
plt.show()

mosaic(data=df, index=['GENDER','OUTCOME'],gap=0.02,title='The Connection Between Gender to Insurance Claim',
       labelizer=lambda k:{('male','0'):108,('male','1'):102,('female','0'):101,('female','1'):66}[k])
plt.show()

mosaic(data=df, index=['DRIVING_EXPERIENCE','OUTCOME'],gap=0.02,title='The Connection Between Driving Experience to Insurance Claim',
       labelizer=lambda k:{('0-9y','0'):38,('0-9y','1'):120,('10-19y','0'):78,('10-19y','1'):39,
                           ('20-29y','0'):64,('20-29y','1'):7,('30y+','0'):26,('30y+','1'):1}[k])
plt.show()

mosaic(data=df, index=['EDUCATION','OUTCOME'],gap=0.02,title='The Connection Between Education Experience to Insurance Claim',
       labelizer=lambda k:{('none','0'):28,('none','1'):48,('high school','0'):91,('high school','1'):71,
                           ('university','0'):90,('university','1'):49}[k])
plt.show()

mosaic(data=df, index=['INCOME','OUTCOME'],gap=0.02,title='The Connection Between Income Experience to Insurance Claim',
       labelizer=lambda k:{('poverty','0'):16,('poverty','1'):59,('middle class','0'):57,('middle class','1'):35,
                           ('working class','0'):20,('working class','1'):42, ('upper class','0'):116,('upper class','1'):32}[k])
plt.show()

mosaic(data=df, index=['VEHICLE_OWNERSHIP','OUTCOME'],gap=0.02,title='The Connection Between Vehicle Owenership Experience to Insurance Claim',
       labelizer=lambda k:{('0','0'):"not owner and no claim - 39",('0','1'):"not owner and claim - 99",('1','0'):"owner and no claim - 170",('1','1'):"owner and claim - 69",}[k])
plt.show()

mosaic(data=df, index=['VEHICLE_YEAR','OUTCOME'],gap=0.02,title='The Connection Between Vehicle Year to Insurance Claim',
       labelizer=lambda k:{('before 2015','0'):172,('before 2015','1'):150,('after 2015','0'):77,('after 2015','1'):18,}[k])
plt.show()

mosaic(data=df, index=['MARRIED','OUTCOME'],gap=0.02,title='The Connection Between Married to Insurance Claim',
       labelizer=lambda k:{('0','0'):"not married and no claim - 84",('0','1'):"not married and claim - 113",('1','0'):"married and no claim - 125",('1','1'):"married and claim - 55",}[k])
plt.show()

mosaic(data=df, index=['CHILDREN','OUTCOME'],gap=0.02,title='The Connection Between Children to Insurance Claim',
       labelizer=lambda k:{('0','0'):"not children and no claim - 54",('0','1'):"not children and claim - 77",('1','0'):"children and no claim - 155",('1','1'):"children and claim - 91",}[k])
plt.show()

mosaic(data=df, index=['AGE','DRIVING_EXPERIENCE'],gap=0.02,title='The Connection Between Postal Code to Insurance Claim')
plt.show()


## union category
mosaic(data=df, index=['AGE','OUTCOME'],gap=0.02,title='The Connection Between Age to Insurance Claim',
       labelizer=lambda k:{('16-25','0'):11,('16-25','1'):69,('26-39','0'):67,('26-39','1'):61,
       ('40-64','0'):85,('40-64','1'):26,('65+','0'):46,('65+','1'):12}[k])
plt.show()

mosaic(data=df, index=['POSTAL_CODE','OUTCOME'],gap=0.02,title='The Connection Between Postal Code to Insurance Claim',
       labelizer=lambda k:{('10238','0'):158,('10238','1'):99,('21217','0'):0,('21217','1'):5,
                           ('32765','0'):43,('32765','1'):55,('92101','0'):8,('92101','1'):9}[k])
plt.show()


### new varible
sns.boxplot(x=df['OUTCOME'], y=df['PAST_ACCIDENTS'], palette="Set3")
plt.xlabel('insurance Claim')
plt.ylabel('past accidents')
plt.title('The Connection Between Past Accidents to Insurance Claim')

sns.boxplot(x=df['OUTCOME'], y=df['SPEEDING_VIOLATIONS'], palette="Set3")
plt.xlabel('insurance Claim')
plt.ylabel('speeding violations')
plt.title('The Connection Between Speeding Violations to Insurance Claim')

## new data table

df_new = pd.read_excel('CarInsurance_Xy_train_new.xlsx')




























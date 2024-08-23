import pandas as pd
import matplotlib.pyplot as plt
import country_converter as coco
#https://pypi.org/project/country-converter/
import seaborn as sns
import pycountry_convert as pc

# Load the datasets
tax_income = pd.read_csv('.../oecd_tax_income.csv')
tax_corporate = pd.read_csv('.../oecd_corporate_profit.csv')
tax_property = pd.read_csv('.../oecd_tax_on_property.csv')
unemployment = pd.read_csv('.../oecd_unemployment.csv')

# Remove average measurements
def remove_averages(df):
    dump = ['OAVG', 'EU27_2020', 'EU28', 'OECD', 'EA19', 'G-7'] 
    return df[~df['LOCATION'].isin(dump)]

tax_income = remove_averages(tax_income)
tax_corporate = remove_averages(tax_corporate)
tax_property = remove_averages(tax_property)
unemployment = remove_averages(unemployment)


tax_income['TIME'] = pd.to_datetime(tax_income['TIME'], format='%Y')
tax_corporate['TIME'] = pd.to_datetime(tax_corporate['TIME'], format='%Y')
tax_property['TIME'] = pd.to_datetime(tax_property['TIME'], format='%Y')


# Drop the 'Flag Codes' column
tax_income = tax_income.drop(columns=['Flag Codes'])
tax_corporate = tax_corporate.drop(columns=['Flag Codes'])
tax_property = tax_property.drop(columns=['Flag Codes'])
unemployment = unemployment.drop(columns=['Flag Codes'])
print(tax_income.head())


#print(tax_income.describe())
# Select countries for plotting
selected_countries = ['GRC', 'AUS', 'DEU', 'USA', 'JPN']

# Filter data for selected countries
filtered_tax_income = tax_income[tax_income['LOCATION'].isin(selected_countries)]

# Plot
plt.figure(figsize=(12, 6))
for country in selected_countries:
    country_data = filtered_tax_income[filtered_tax_income['LOCATION'] == country]
    plt.plot(country_data['TIME'], country_data['Value'], label=coco.convert(names=country, to='name_short'))

plt.title('Tax on Personal Income Across Years')
plt.xlabel('Year')
plt.ylabel('Tax on Personal Income (%)')
plt.legend()
plt.show()

#For full dataset
plt.figure(figsize=(12, 6))
sns.histplot(tax_income['Value'], kde=False)
plt.title('Distribution of Tax on Personal Income')
plt.xlabel('Tax on Personal Income (%)')
plt.ylabel('Frequency')
plt.show()

def country_to_continent(country_code):
    try:
        # Perform custom continent mapping
        if country_code in ['CAN', 'USA', 'MEX', 'BRA', 'ARG', 'COL', 'PER', 'CHL', 'VEN', 'ECU', 'GTM', 'CUB', 'BOL', 'DOM', 'HND', 'PRY', 'NIC', 'SLV', 'HTI', 'CRI', 'PAN', 'URY', 'JAM', 'TTO', 'BRB', 'BHS', 'SUR', 'BLZ', 'GUY', 'GRD', 'ATG', 'DMA', 'KNA', 'VCT', 'LCA', 'TCA', 'AIA', 'MSR', 'CUW', 'SXM', 'BES']:
            return 'America'  # Group North and South America into one continent
        elif country_code in ['AUS', 'NZL', 'PNG']:
            return 'Asia'  # Group Australia into Asia
        else:
            # Use pycountry-convert to get the continent name for other countries
            country_alpha2 = coco.convert(names=country_code, to='ISO2')
            continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
            continent_name = pc.convert_continent_code_to_continent_name(continent_code)
            return continent_name
    except KeyError:
        # Handle KeyError (invalid country code) gracefully
        return country_code  # or any default value you prefer
# Function to get continent from country code

# Add a continent column to each DataFrame
tax_income['CONTINENT'] = tax_income['LOCATION'].apply(country_to_continent)
#tax_corporate['CONTINENT'] = tax_corporate['LOCATION'].apply(country_to_continent)
#tax_property['CONTINENT'] = tax_property['LOCATION'].apply(country_to_continent)
#unemployment['CONTINENT'] = unemployment['LOCATION'].apply(country_to_continent)

# Verify the results
#print(tax_income[['LOCATION','CONTINENT']].head())


plt.figure(figsize=(12, 6))
sns.histplot(data=tax_income, x='Value', hue='CONTINENT', multiple='stack', kde=True)
plt.title('Distribution of Tax on Personal Income by Continent')
plt.xlabel('Tax on Personal Income (%)')
plt.ylabel('Frequency')
plt.show()


corporate_tax_statistics = tax_corporate.groupby('LOCATION')['Value'].agg(['mean', 'median']).reset_index()
corporate_tax_statistics.columns = ['Country', 'Average Tax on Corporate Profits', 'Median Tax on Corporate Profits']
print(corporate_tax_statistics)

# Merge datasets on Country and Year
merged_taxes = pd.merge(tax_income, tax_property, on=['LOCATION', 'TIME'], suffixes=('_income', '_property'))

# Calculate the ratio
merged_taxes['Tax Ratio'] = merged_taxes['Value_income'] / merged_taxes['Value_property']

# Filter data for selected countries
filtered_ratio = merged_taxes[merged_taxes['LOCATION'].isin(selected_countries)]

# Plot
plt.figure(figsize=(12, 6))
for country in selected_countries:
    country_data = filtered_ratio[filtered_ratio['LOCATION'] == country]
    plt.plot(country_data['TIME'], country_data['Tax Ratio'], label=coco.convert(names=country, to='name_short'))

plt.title('Income to Property Tax Ratio Across Years')
plt.xlabel('Year')
plt.ylabel('Income to Property Tax Ratio')
plt.legend()
plt.show()


unemployment['TIME'] = pd.to_datetime(unemployment['TIME'], format='%Y-%m')

# Extracting only the year and replacing the 'TIME' column with it
unemployment['YEAR'] = unemployment['TIME'].dt.year
unemployment_rate = unemployment.groupby(['LOCATION', 'YEAR'])['Value'].mean().reset_index()
unemployment_rate.columns = ['Country', 'Year', 'Unemployment Rate']
print(unemployment_rate)

unemployment['CONTINENT'] = unemployment['LOCATION'].apply(country_to_continent)

unemployment_rate_continent = unemployment.groupby(['CONTINENT', 'YEAR'])['Value'].mean().reset_index()
unemployment_rate_continent.columns = ['Continent', 'Year', 'Unemployment Rate']
print(unemployment_rate_continent)

def standardize_unemployment(df):
    # Select numerical column ('Value' column containing unemployment rate)
    numeric_column = 'Value'
    
    # Standardize unemployment rate values
    standardized_values = (df[numeric_column] - df[numeric_column].mean()) / df[numeric_column].std()
    
    # Merge standardized unemployment rate values with non-numerical columns
    standardized_df = pd.concat([df.drop(columns=numeric_column), standardized_values], axis=1)
    
    return standardized_df

# Assuming 'unemployment' DataFrame contains 'CONTINENT' column and other relevant columns
# Group by 'CONTINENT' column and apply standardization function
standardized_unemployment_per_continent = unemployment.groupby('CONTINENT').apply(standardize_unemployment)

# Print the standardized DataFrame
print(standardized_unemployment_per_continent)


# Filter for years 2011-2015
unemployment_filtered = unemployment[(unemployment['YEAR'] >= 2011) & (unemployment['YEAR'] <= 2015)]

# Calculate the average unemployment rate per country
avg_unemployment = unemployment_filtered.groupby('LOCATION')['Value'].mean().reset_index()
avg_unemployment.columns = ['Country', 'Average Unemployment Rate']

# Rank countries in decreasing order
ranked_unemployment = avg_unemployment.sort_values(by='Average Unemployment Rate', ascending=False).reset_index(drop=True)
print(ranked_unemployment)
















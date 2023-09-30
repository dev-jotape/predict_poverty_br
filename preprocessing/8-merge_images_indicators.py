import pandas as pd
from os import listdir
from os.path import isfile, join

### IMPORT POPULATION DENSITY FILE (CENSO 2022) ---------------------------------------------------
population_density = pd.read_excel('../excel-files/population_density_censo2022.xlsx', skiprows=3, skipfooter=1)
population_density.columns = ['city_code', 'city', 'density']
# print(population_density.info())

### IMPORT POPULATION FILE (CENSO 2022) ---------------------------------------------------
raw_population = pd.read_excel('../excel-files/previa_populacao_censo_2022.xls', skiprows=1, skipfooter=34, converters={'COD. MUNIC':str,'COD. UF':str})
raw_population['city_code'] = (raw_population['COD. UF'] + raw_population['COD. MUNIC']).astype(int)

population = raw_population[['city_code', 'NOME DO MUNICÍPIO', 'UF', 'POPULAÇÃO']]
population.columns = ['city_code', 'city_name', 'city_uf', 'population']

def formatPopulation(x):
    if (type(x) != int):
        pop = x.split('(')[0]
        pop = pop.replace('.','')
        return int(pop)
    else:
        return int(x)
population['population'] = population.apply(lambda x: formatPopulation(x['population']), axis=1)
# print(population)

### IMPORT INCOME PROJECTION FILE (FGV) ---------------------------------------------------
income = pd.read_excel('../excel-files/projecao_renda_fgv_2021.xlsx', skiprows=8, skipfooter=1, converters={'R$':float})
income = income[['Município', 'UF', 'R$']]
income.columns = ['city_name', 'city_uf', 'income']

income.drop(index=income.index[0], axis=0, inplace=True)
income['city_name'] = income['city_name'].str.split('/', expand=True)[0]
# print(income)

### IMPORT IMAGES PATHS ---------------------------------------------------

class1 = ['dataset/google_images/class1/' + f for f in listdir('../dataset/google_images/class1/') if isfile(join('../dataset/google_images/class1/', f))]
class2 = ['dataset/google_images/class2/' + f for f in listdir('../dataset/google_images/class2/') if isfile(join('../dataset/google_images/class2/', f))]
class3 = ['dataset/google_images/class3/' + f for f in listdir('../dataset/google_images/class3/') if isfile(join('../dataset/google_images/class3/', f))]

allFiles = class1 + class2 + class3

imagePaths=pd.DataFrame(allFiles, columns=['path']) 

def split_city(x):
    path = x.split('/')[3]
    city_code = path.split('_')[0]
    rank = path.split('_')[1]
    return [int(city_code), int(rank)]

imagePaths[['city_code', 'rank']] = imagePaths.apply(lambda x: split_city(x.path),  axis='columns', result_type='expand')
# print(imagePaths)

### MERGE FILES ---------------------------------------------------

pop_income = pd.merge(population, income, on=['city_name', 'city_uf'])
pop_income_density = pd.merge(pop_income, population_density, on='city_code')
final_merge = pd.merge(imagePaths, pop_income_density, on='city_code')
# print(final_merge)
# print(final_merge.info())

final_merge.to_csv('../excel-files/cities_indicators.csv', index=False)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel('asset-v1_SkillFactory+MIFIML-2sem+2025+type@asset+block@Данные_для_курсовои__Классическое_МО.xlsx')

# размерность и типы данных 
df.info()
# оцениваем базовую статистику 
print(df.describe()) 
print(df.head()) 
print(f"Исходная размерность датасета: {df.shape}")

# удаление технического столбца 'Unnamed: 0'
df.drop(columns=['Unnamed: 0'], inplace=True)
print(f"Размерность после удаления технического индекса: {df.shape}")

# поиск и удаление дубликатов строк
initial_duplicates = df.duplicated().sum()
print(f"Обнаружено полных дубликатов строк: {initial_duplicates}")

# если они есть
if initial_duplicates > 0:
    df.drop_duplicates(inplace=True)
    print(f"Размерность после удаления дубликатов: {df.shape}")

# проверка на наличие пропущенных значений (NaN)
total_missing = df.isnull().sum().sum()
print(f"Общее количество пропущенных значений в датасете: {total_missing}")

# обработка пропущенных значений
df.fillna(df.median(), inplace=True)
print(f"Пропусков после обработки: {df.isnull().sum().sum()}")

# поиск и удаление константных признаков (Variance Threshold)
constant_columns = [col for col in df.columns if df[col].nunique() <= 1]
print(f"Обнаружено константных признаков: {len(constant_columns)}")
print(constant_columns)
df.drop(columns=constant_columns, inplace=True)
print(f"Размерность после удаления констант: {df.shape}")

#  список наших целевых переменных
targets = ['IC50, mM', 'CC50, mM', 'SI']

#  распределения  
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, col in enumerate(targets):
    sns.histplot(df[col], bins=50, ax=axes[i], kde=True, color='red')
    axes[i].set_title(f'Исходное распределение\n{col}')
plt.tight_layout()
plt.show()

#  логарифмирование (используем np.log1p = ln(1+x)
for col in targets:
    df[f'{col}_log'] = np.log1p(df[col])

#  распределения после трансформации
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, col in enumerate(targets):
    sns.histplot(df[f'{col}_log'], bins=50, ax=axes[i], kde=True, color='green')
    axes[i].set_title(f'После логарифмирования\n{col}_log')
plt.tight_layout()
plt.show()

# получаем метрику асимметрии 
print("\n Коэффициенты асимметрии (Skewness) ")
for col in targets:
    skew_before = df[col].skew()
    skew_after = df[f'{col}_log'].skew()
    print(f"{col}: было {skew_before:.2f} | стало {skew_after:.2f}")

# анализ корреляций, мультиколлинеарность
targets_all = ['IC50, mM', 'CC50, mM', 'SI', 'IC50, mM_log', 'CC50, mM_log', 'SI_log']
features = df.drop(columns=targets_all)

# матрица  Пирсона
corr_matrix = features.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# признаки, у которых корреляция друг с другом больше 0.90
to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
print(f"Найдено сильно скоррелированных признаков (дубликатов): {len(to_drop)}")
df.drop(columns=to_drop, inplace=True)
print(f"Размерность данных после очистки коррелятов: {df.shape}")

# проверка логики SI
df['SI_calculated'] = df['CC50, mM'] / df['IC50, mM']

#  средняя ошибка между табличным SI и расчетным 
diff = (df['SI_calculated'] - df['SI']).abs().mean()
print(f"Средняя разница между расчетом и таблицей: {diff:.5f}")
df.drop(columns=['SI_calculated'], inplace=True)

df.to_csv('cleaned_molecular_data.csv', index=False)
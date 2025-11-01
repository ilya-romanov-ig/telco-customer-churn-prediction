# telco-customer-churn-prediction

### Цель проекта
Построить модель, способную предсказать возможный отток клиентов.

### Данные
Проект выполнен на основе датасета [Kaggle: IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). Датасет состоит из 7043 строк и 21 столбца (id пользователя, 4 числовых признака, 16 категориальных признаков).

### Стек
- Python
- Pandas
- Scikit-learn
- Imblearn
- Matplotlib
- Seaborn
- joblib
- SHAP
- CatBoost
- XGBoost

### Структура репозитория
- data - хранение csv файлов
- model - хранилище моделей
- notebooks - исследование данных, построение моделей
- src - скрипты

### Инструкция по запуску
1) ```pip install -r requirements.txt```.
2) Перейти в папку src: ```cd src```.
3) Запустить препроцессинг данных: ```python process.py```.
4) Запустить создание и обучение модели: ```python train.py```.
5) Запустить создание предсказания: ```python predict.py```. Опции: ```--input data_file_name.csv``` - выбор файла для загрузки в модель, ```--output data_file_name.csv``` - указание расположения файла для предсказаний.

### Методология
1) EDA - разведочный анализ датасета на пропущенные значения, выбросы, грязные данные.
2) Feature Engineering - валидация числовых признаков, создание Label Encoding и One-Hot Encoding для категориальных признаков.
3) Подбор baseline моделей, борьба с дисбалансом классов при помощи SMOTE.
4) Подбор гиперпараметров для baseline моделей при помощи RandomizedSearchCV, показавших наилучшие результаты "из коробки".
5) Глубокая интерпретация результатов лучшей модели (Precision-Reacall Curve, ROC Curve). Использование SHAP для определения влияния параметров модели на итоговый результат.

### Результаты
#### Результаты исследования baseline моделей
В качестве базовых моделей были выбраны: Logistic Regression, Random Forest, Gradient Boosting, SVC, CatBoost, XGBoost

Качественный анализ базовых моделей:
| Model | Accuracy | ROC-AUC | F1 | Recall |
| :---: | :---: | :---: | :---: | :---: |
| Logistic Regression| 0.737 | 0.847 | 0.612 | 0.783 |
| Random Forest | 0.765 | 0.827 | 0.576 | 0.602 |
| Gradient Boosting | 0.779 | 0.841 | 0.624 | 0.693 |
| SVC | 0.756 | 0.817 | 0.606 | 0.706 |
| CatBoost | 0.779 | 0.836 | 0.596 | 0.615 |
| XGBoost | 0.776 | 0.814 | 0.580 | 0.583 |

#### Подбор гиперпараметров
Для дальнейшего исследования были выбрани следующие модели:
- Random Forest
- CatBoost
- XGBoost

После проведения анализа гиперпараметров выбранных моделей получены следующие результаты:
| Model | Accuracy | F1 | Recall | ROC-AUC |
| :---: | :---: | :---: | :---: | :---: |
| Random Forest | 0.828 | 0.830 | 0.850 | 0.910 |
| CatBoost | 0.826 | 0.804 | 0.809 | 0.934 |
| XGBoost | 0.829 | 0.818 | 0.830 | 0.926 |

По результатам анализа была выбрана Random Forest со следующими гиперпараметрами:
- n_estimators: 250
- min_samples_split: 12
- min_samples_leaf: 3
- max_features: sqrt
- max_depth: 20
- criterion: gini



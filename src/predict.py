import pandas as pd
import joblib
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='../data/test_preprocessed.csv', help='Путь для CSV файла с данными для предсказания')
    parser.add_argument('--model', type=str, default='../model/trained_model_pipeline.pkl')
    parser.add_argument('--output', type=str, required=True, help='Путь для выходного CSV файла')

    args = parser.parse_args()
    print('Загрузка модели...')
    try:
        model_pipeline = joblib.load(args.model)
    except Exception as e:
        print('Ошибка: ', e)
        return

    print('Загрузка данных...')
    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print('Ошибка загрузки: ', e)
        return

    preds = model_pipeline.predict(df)
    probas = model_pipeline.predict_proba(df)

    res_df = pd.DataFrame({
        'preds': preds,
        'probas': probas[:, 1]
    })
    print(f'Сохранение результатов в {args.output}')
    try:
        res_df.to_csv(args.output, index=False)
    except Exception as e:
        print('Ошибка сохранения предсказаний. Сохранение в стандартный путь: ../output.csv')
        res_df.to_csv('..output.csv', index=False)

    print('Готово!')

if __name__ == '__main__':
    main()
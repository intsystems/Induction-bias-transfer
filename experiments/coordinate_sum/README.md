# Инвариантность MLP к перестановке координат

Базовый эксперимент с известной аугментацией: случайной перестановкой координат.

Координаты входа независимо генерируются из стандартного нормального
распределения. Целевое значение — `sin(sum(x) / sqrt(dim)).

MLP с двумя скрытыми слоями ReLU обучается с функцией потерь:

```text
loss = mean((f(x) - y)^2) + weight * mean(abs(f(x) - f(shuffle(x))))
```

По умолчанию: 256 train-примеров, 4096 test-примеров, 20 координат,
два скрытых слоя по 64 нейрона, 100 эпох, веса штрафа `0`, `0.1`, `0.5`, `1`.

## Запуск

Зависимости: `torch`, `matplotlib`. Команды из корня репозитория:

```bash
python experiments/coordinate_sum/experiment.py --output results/coordinate_sum/metrics.json
python experiments/coordinate_sum/plot_results.py results/coordinate_sum/metrics.json --output results/coordinate_sum/curves.png
```

Без `--output` пути по умолчанию — `results/metrics.json` для обучения
и `results/curves.png` для графиков, относительно текущей рабочей директории.

JSON содержит конфигурацию и историю каждой модели: обучающий MSE и штраф,
тестовый MSE на исходных и перемешанных входах, `invariance_mae` — среднее
`abs(f(x) - f(shuffle(x)))` на тесте. Тестовые метрики включают эпоху 0.


Пример изменения параметров:

```bash
python experiments/coordinate_sum/experiment.py --device cpu --train-size 128 --weights 0 0.01 0.1 1 --seed 1 --output results/coordinate_sum/seed1.json
```

## Гипотеза

Штраф улучшает инвариантность и может улучшить обобщение на малой выборке.
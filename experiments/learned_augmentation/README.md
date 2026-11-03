# Перенос обученной аугментации

Для каждого учителя CNN/MLP обучаем распределение по четырём сдвигам
последовательности, затем используем его при обучении MLP на новых шаблонах.

Генератор из `assets/conv_data_v2.ipynb`. Зависимости: `torch`, `matplotlib`. Команды из корня проекта:

```bash
python experiments/learned_augmentation/experiment.py
python experiments/learned_augmentation/plot_results.py
```

Учитель обучается на исходной задаче и замораживается. Аугментация — распределение
`p = softmax(a)` по сдвигам на один элемент: вправо/влево с заполнением `-1`/`+1`.

```text
policy_loss = E_x E_(k~p) [(sigmoid(T(x)) - sigmoid(T(shift_k(x)))^2]
              - entropy_weight * H(p)
student_loss = BCE(student(x), y)
               + consistency_weight * E_(k~p) |sigmoid(student(x)) - sigmoid(student(shift_k(x)))|
```

Политика обучается только на исходных train-входах и ответах учителя.
Энтропия уменьшает концентрацию на одном сдвиге. Ожидания по четырём сдвигам
считаются точно. Затем распределение замораживается и переносится на новые задачи
без учителя. Градиенты штрафа студента проходят через оба предсказания.

Сравниваются `baseline` без штрафа, `uniform` с равномерной аугментацией,
`learned_cnn` и `learned_mlp`. Улучшение качества является проверяемой гипотезой.

Сдвиг может изменить метку у границы. Политика не получает фильтр по истинному
шаблону. После обучения в JSON сохраняется диагностика `policy_metrics` на target
test: доля сохранённых меток, доля изменившихся входов, отклонение ответов студента
и accuracy относительно истинных **новых** меток преобразованных входов.
Истинный шаблон используется здесь только для диагностики.

Графики: `sample_efficiency.png` и `curves_n*.png` рядом с JSON.
Линии усредняются по целевым задачам, заливка — ± одно стандартное отклонение
между задачами. Это не доверительный интервал и не разброс по seed.
Accuracy в JSON — доля от 0 до 1. `invariance_mae` в истории измеряется на
фиксированных сохраняющих метку парах генератора. Эпоха 0 — до обучения.

```bash
python experiments/learned_augmentation/experiment.py --device cuda --consistency-weight 0.1 --entropy-weight 0.001 --seed 1 --output results/pattern_augmentation/seed1.json
python experiments/learned_augmentation/plot_results.py results/pattern_augmentation/seed1.json --methods baseline uniform learned_cnn
```
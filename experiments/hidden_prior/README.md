# Скрытые состояния как инициализация и априор

Обучаем encoder MLP воспроизводить скрытые активации учителя CNN/MLP,
затем переносим его веса на задачу с новым шаблоном.

Генератор из `assets/conv_data_v2.ipynb`, модели и обучение находятся в
[`../_pattern_transfer/common.py`](../_pattern_transfer/common.py).
Описание данных: [`../_pattern_transfer/README.md`](../_pattern_transfer/README.md).
Зависимости: `torch`, `matplotlib`. Команды из корня проекта:

```bash
python experiments/hidden_prior/experiment.py
python experiments/hidden_prior/plot_results.py
```

Стадии эксперимента:

1. Обучаем и замораживаем учителя на исходном шаблоне.
2. На исходных train-входах обучаем encoder MLP минимизировать
   `MSE(h_student(x), h_teacher(x) / scale)`. `scale` — RMS активаций учителя
   на исходном train, одинаковый для всех координат.
3. Переносим encoder в целевой MLP, создаём свежую голову и обучаем на метках
   нового шаблона.

У CNN берутся активации ReLU до pooling, развёрнутые в вектор. Ширина скрытого
слоя MLP совпадает с числом этих активаций. Голова исходного классификатора
не переносится. Target test не используется для воспроизведения признаков.

Два режима переноса:

```text
init:  loss = BCE(student(x), y)
prior: loss = BCE(student(x), y) + prior_weight * mean((encoder_params - center)²)
```

`center` — encoder после воспроизведения активаций; штраф не применяется к голове.
Это MAP-подобный штраф с изотропным гауссовским априором на параметры encoder.

Методы: `baseline`, `random_prior` (штраф к случайному начальному encoder),
`cnn_init`, `cnn_prior`, `mlp_init`, `mlp_prior` и соответствующие
`untrained_cnn_*`, `untrained_mlp_*`. Необученные учителя имеют те же начальные
веса, что обученные версии. Контроли позволяют оценить вклад архитектуры,
дополнительного обучения и обученных представлений. В пределах целевой задачи
головы всех студентов имеют одинаковую случайную инициализацию.

В [McCoy & Griffiths (2025)](https://www.nature.com/articles/s41467-025-59957-y)
распределение задач определяет инициализацию через MAML. Здесь проверяется
родственная гипотеза через скрытые состояния учителя; MAML не реализуется.

JSON по умолчанию: `results/pattern_hidden_prior/metrics.json`.
Он содержит конфигурацию, шаблоны, истории учителей, ошибки воспроизведения
признаков на source train/test и истории студентов. Файл обновляется после
подготовки учителей и каждого завершённого целевого запуска.

Графики: `sample_efficiency.png` и `curves_n*.png` рядом с JSON.
Линии усредняются по целевым задачам, заливка — ± одно стандартное отклонение
между задачами. Это не доверительный интервал и не разброс по seed.
Accuracy в JSON — доля от 0 до 1. `invariance_mae` измеряется на фиксированных
сохраняющих метку парах генератора. Эпоха 0 — до обучения.

```bash
python experiments/hidden_prior/experiment.py --device cuda --prior-weight 10 --feature-epochs 100 --seed 1 --output results/pattern_hidden_prior/seed1.json
python experiments/hidden_prior/plot_results.py results/pattern_hidden_prior/seed1.json --methods baseline cnn_init cnn_prior untrained_cnn_init
```

Сравнивайте качество учителя, ошибку воспроизведения признаков, accuracy
и чувствительность к сдвигу. Проверяйте несколько seed и весов штрафа
с отдельными файлами. Целевой test не используется для выбора гиперпараметров.

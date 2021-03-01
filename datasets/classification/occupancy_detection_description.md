# Occupancy Detection Data Set

## Источник
Датасет взят из открытого источника UCI [здесь](https://archive.ics.uci.edu/ml/datasets/Occupancy+Detection+).

## Краткое описание
Эти данные используются для бинарной классификации (заполненность комнаты) по температуре, влажности, освещенности и уровне CO2.  
Фактическая заполненность была получена по фотографиям с отметкой времени, которые делались каждую минуту.   
**Связанные публикации:** [Accurate occupancy detection of an office room from light, temperature, humidity and CO2 measurements using statistical learning models. Luis M. Candanedo, Véronique Feldheim. Energy and Buildings. Volume 112, 15 January 2016, Pages 28-39.](https://www.researchgate.net/profile/Luis_Candanedo_Ibarra/publication/285627413_Accurate_occupancy_detection_of_an_office_room_from_light_temperature_humidity_and_CO2_measurements_using_statistical_learning_models/links/5b1d843ea6fdcca67b690c28/Accurate-occupancy-detection-of-an-office-room-from-light-temperature-humidity-and-CO2-measurements-using-statistical-learning-models.pdf?origin=publication_detail)

## Целевая переменная
**Суть задачи:** определить, заполнена команта или нет.  
**Целевой признак:** Occupancy, заполненность, 0 или 1, 0 -- не заполнена, 1 -- заполнена

## Признаки
| Признак | Описание | Тип |
|:-:|:-:|:-:|
|Date Time|Дата и время измерений|Дата/Время|
|Temperature|Температура в помещении|Вещественный|
|Relative Humidity|Относительная влажность в помещении|Вещественный|
|Light|Освещённость в помещении|Вещественный|
|CO2|Уровень углекислого газа в помещении|Вещественный|
|Humidity Ratio|Коэффициент влажности в помещении|Вещественный|
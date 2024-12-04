# Определение статьи ДДС по назначению платежа

Сервис для определния статьи ДДС в поступлении на расчетный счет, списании с расчетного счета: по виду операции и назначению платежа

```
[POST] /train - form-data:
 file - File - Excel (xlsx) для обучения (например за предыдущий закрытый квартал) с колонкаим:
    ВидОперации, НазначениеПлатежа, СтатьяДвиженияДенежныхСредств
 model_id - string - текстовый идентификатор модели.
    Например ИНН организации, подразделения, год (обычно статьи ДДС утверждаются ежегодно)
 [ANSWER] - JSON - с полем message (Model trained successfully)

[POST] /test - form-data:
 file - File - Excel (xlsx) для тестирования (например за текущий месяц) с колонкаим:
    ВидОперации, НазначениеПлатежа, СтатьяДвиженияДенежныхСредств
 model_id - string - текстовый идентификатор модели.
    Например ИНН организации, подразделения, год (обычно статьи ДДС утверждаются ежегодно)
 [ANSWER] - JSON - с полем accuracy "точность" модели

[POST] /predict - form-data:
 model_id - Text - имя обученной ранее модели
 operataion - string - вид операции.
    Например Оплата от покупателя, Прочее поступление, Оплата поставщику
 text - string - назначение платежа
 [ANSWER] - JSON - с полем article (подходящий текст из колонки СтатьяДвиженияДенежныхСредств)

[POST] /stringmatch - json:
 string_list - string[] - номера договоров, счетов;
    более длинные строки имеют бОльший приоритет даже не смотря на полное совпадение меньшего текста
	Например "Счет № 02" > "01" в тексте "Оплата по счету 02 от 01.01.2024"
 text - string - назначение платежа
 [ANSWER] - JSON - с полем prediction (подходящий текст из string_list)
```

## Запуск
Построить имидж (однократно)
`docker build --tag textclassify .`

Запустить
`docker run --name article -p 5050:5050 textclassify`

Сохранить из 1С результат запроса в XLSX за закрытый период (месяц, квартал)
```
ВЫБРАТЬ
	ТабРеквизиты.ВидОперации КАК ВидОперации,
	ТабРеквизиты.НазначениеПлатежа КАК НазначениеПлатежа,
	ТабРеквизиты.СтатьяДвиженияДенежныхСредств КАК СтатьяДвиженияДенежныхСредств
ИЗ Документ.ПоступлениеНаРасчетныйСчет КАК ТабРеквизиты
ГДЕ ТабРеквизиты.Дата МЕЖДУ &ДатаНачала И &ДатаОкончания

ОБЪЕДИНИТЬ ВСЕ

ВЫБРАТЬ
	ТабРеквизиты.ВидОперации,
	ТабРеквизиты.НазначениеПлатежа,
	ТабРеквизиты.СтатьяДвиженияДенежныхСредств
ИЗ Документ.СписаниеСРасчетногоСчета КАК ТабРеквизиты
ГДЕ ТабРеквизиты.Дата МЕЖДУ &ДатаНачала И &ДатаОкончания
```

Обучить модель `/train`; Использовать и радоваться: `/predict`

Пример использования `onec_example` - выгрузка обработки (файлы в формате 1С) с примером использования.
В обработке используется [КоннекторHTTP](https://github.com/vbondarevsky/Connector) Владимира Бондаревского.

Примеры из postman:
![image](https://github.com/ilya2184/TextClassify/assets/14931660/8a93ec21-f530-43df-aa5b-7fa0faf20768)
![image](https://github.com/ilya2184/TextClassify/assets/14931660/180340a6-035f-4896-b7e8-fc84ad98605e)


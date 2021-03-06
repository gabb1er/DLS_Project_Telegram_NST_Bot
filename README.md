# DLS_Project_Telegram_NST_Bot
* Данный репозиторий представляет собой Финальный проект в Школе Глубокого Обучение МФТИ.
* Код, представленный здесь запущен на AWS EC2 intsance. Бот работает автономно 24/7: @dls_project_nst_bot
* Все вычисления выполняются на ЦПУ, поэтому процесс переноса стиля на изображение занимет 3-5 минут.
* Бот поддерживает команду /help со всей необходимой информацией для использования.

## tg_bot.py
Данный файл содержит основной код бота: хендлеры для обработки сообщений пользователя и диалог для запуска переноса стиля и вывода результата пользователю.

## bot_messages.py
В данном файле собраны все текстовые сообщения, которые бот отправляет пользователю.

## style_transfer.py
Модуль в котором реализована сверточная модель для переноса стиля. Модель основана на 5ти верхних свертках VGG-19, обученной на IMAGANET. Т.к. сама VGG-19 весит около 500 Мб, то в данном модуле также реализован класс для того, чтобы воспроизводить верхние слои данной модели и загружать предобученные веса только лишь для них.

## config.py
В данном файле должен находится ключ авторизации бота (удален по соображениям безопасности).

## VGG19_Head.pth
Предобученные веса для верхних слоев VGG-19.

## Example Images
Следующие изображения представляют собой пример работы модели и отправляются ботом по команде /example:
* example_content_img.jpg
* example_style_img.jpg
* example_output_img.jpg

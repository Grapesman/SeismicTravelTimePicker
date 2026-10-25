# Обновление SeisPeak до v0.4.0 через GitHub (только браузер)

1. Откройте репозиторий **Grapesman/SeismicTravelTimePicker** → **Code** → **Add file** → **Upload files**.
2. Перетащите файлы из папки `SeismicTravelTimePicker` обновлённого архива. Для изменённых файлов GitHub покажет замену. Если браузер не захватит скрытую `.github`, обновите workflow отдельно (шаг 4).
3. Напишите сообщение `Release v0.4.0: multi-SGY, zoom, gain controls, time units` и нажмите **Commit changes**.
4. В репозитории откройте `.github/workflows/windows.yml`, нажмите значок **карандаша** (Edit), замените содержимое одноимённым файлом из архива и нажмите **Commit changes**. Убедитесь, что в файле версия `v0.4.0`, а не `v0.3.0`.
5. Откройте **Actions** → **Windows EXE — Build and Release**. Убедитесь, что автоматическая сборка после изменения файлов закончилась успешно.
6. Нажмите **Run workflow** → выберите `main`, поставьте флажок **Publish or update v0.4.0 in Releases with Windows EXE**, затем **Run workflow**.
7. Проверьте **Releases → v0.4.0 → Assets** и скачайте `SeismicTravelTimePicker.exe`. Протестируйте приложение в Windows перед распространением.

**Важно:** GitHub не распаковывает загруженный ZIP автоматически. Загрузите файлы/папки из архива. Не добавляйте исходные SGY, `.venv` или сохранённые пользовательские сеансы.


**Брендинг:** логотип и значок добавлены в `assets/`. Проверьте, что эта папка попала на GitHub. Windows EXE получит свою иконку только после обновления `.github/workflows/windows.yml` из архива.

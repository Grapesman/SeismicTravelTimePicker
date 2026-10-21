# Как опубликовать программу на GitHub (без терминала)

## 1. Разместить исходники

1. Установите GitHub Desktop (https://desktop.github.com/) и войдите в свою учётную запись GitHub.
2. Откройте GitHub Desktop → **File → Add local repository…** → **Choose…**, выберите папку `SeismicTravelTimePicker` из этого архива.
3. Если GitHub Desktop предложит **Create a repository here**, нажмите эту кнопку. Имя проекта — `SeismicTravelTimePicker`, затем **Create repository**.
4. Если файлы не были добавлены в первый коммит автоматически, в левой панели **Changes** отметьте файлы, в поле Summary напишите `Initial release v0.3.0`, нажмите **Commit to main**.
5. Нажмите **Publish repository**. Снимите галочку **Keep this code private**, если репозиторий должен быть общедоступен. Подтвердите **Publish repository**.

**Важно:** перед публикацией убедитесь, что в репозитории нет реальных `.sgy/.segy`, личных данных, паролей и файлов `.venv`.

## 2. Получить Windows EXE

1. Откройте опубликованный репозиторий в браузере: GitHub Desktop → **Repository → View on GitHub**.
2. Откройте вкладку **Actions**, найдите **Windows EXE — Build and Release**.
3. При первом запуске GitHub может предложить включить Actions — подтвердите, если доверяете опубликованному файлу `.github/workflows/windows.yml`.
4. После публикации `main` рабочий процесс должен автоматически начать сборку. Дождитесь зелёной галочки в Actions. Если сборка не запустилась, нажмите **Run workflow**, оставив **Publish or update...** выключенным.
5. Если сборка успешна, откройте **Actions → Windows EXE — Build and Release → Run workflow**. Поставьте галочку **Publish or update v0.3.0 in Releases with Windows EXE**, нажмите зелёную кнопку **Run workflow**.
6. Когда оба задания завершатся зелёным цветом, на странице репозитория справа откройте **Releases → v0.3.0**. В **Assets** появится `SeismicTravelTimePicker.exe`.
7. Скачайте EXE и проверьте на настоящем Windows 10/11 или Windows в Parallels. Сам факт успешной сборки не заменяет проверку интерфейса и работы с SGY.

Если второе задание выдаст ошибку `Resource not accessible by integration`, проверьте **Settings → Actions → General → Workflow permissions**. Для записи релиза требуется `Read and write permissions` у токена GitHub Actions; изменить этот параметр можно только если разрешают настройки владельца/организации. Повторите запуск публикации.

## 3. Что дать пользователям

- Ссылку на страницу **Releases** опубликованного репозитория.
- Скачивать нужно именно EXE из раздела **Assets**, а не `Source code (zip)`.
- Краткое руководство по кнопкам уже находится в корневом `README.md`.

## 4. При следующем выпуске

Перед выпуском новой версии измените номер версии в `seismic_picker/__init__.py`, имя релиза в `.github/workflows/windows.yml`, а также заголовок `RELEASE_NOTES.md`. Выполняйте сборку и публикацию только после тестирования изменений.

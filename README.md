# pug-optimization-1
Repository for lab 1 of ITMO Optimization Methods course

Правила работы в общем репозитории:
1. Когда добавляешь что-то новое создавай ветку `(git checkout -b <branch-name>)`, а уже затем вливай её в `main` `(git checkout main && git pull && git merge <branch-name>)`
2. Если при вливании произошли конфликты, лучше написать в чат или сделать PR (можно игнорировать, если ты уверен, что ничего не сломаешь)
3. Have fun!

О структуре:
- В папку (common)[common] кладём важные общие штуки типа интерфейсов, она не должна часто меняться
- В папку (optimize)[optimize] кладём всё, что относится к созданным методам оптимизации (таск - разработка)
- В папку (research)[research] кладём скрипты, запускающие оптимизаторы на разных данных и показывающие разницу между ними

To be done:
- улучшение интерфейсов
- скрипт, перегенерирующий все графики и данные из research
- ci/cd
